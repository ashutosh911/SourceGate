"""Multi-negative null-normalized evidence experiment.

This is the confirmatory successor to the single-negative CEG pilot.  For each
query and source, it estimates a source-format null distribution using contexts
retrieved for semantically distant queries:

    gain_s(q) = log p(q | c_s(q)) - mean_m log p(q | c_s(q^-_m))
    z_s(q)    = gain_s(q) / pooled_null_sd_s

Donors are selected without using the target routing label or gold answer.  We
first restrict to the least-similar quartile of query embeddings, then prefer
contexts whose token length is at least the target length (so exact matching
usually requires truncation rather than artificial padding).

The raw positive likelihoods are reused from the completed script_39 pilot.

Usage (GPU; activate the chestx conda environment first):
  python script_40_null_normalized_evidence.py --per-class 100 --num-negatives 5 --resume
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import script_12_prefrag_conf as S12
import script_39_counterfactual_evidence_gain as S39


OUT_DIR = Path("phase9_results")
SOURCE_TYPES = S39.SOURCE_TYPES
SEED = S39.SEED
READERS = {
    "llama": (S12.LLM_NAME, ""),
    # This is a deliberately different-family stress test.  It is math-tuned,
    # so it is evidence about mechanism portability, not a new general baseline.
    "qwen_math": ("Qwen/Qwen2.5-Math-7B-Instruct", "qwenmath"),
}


def distant_donors(i, source, selected, embs, lengths, num_negatives):
    """Choose label-free, semantically distant, approximately length-matched donors."""
    pool = np.asarray([j for j in selected if j != i], dtype=int)
    # Embeddings are expected to be normalized, but normalize defensively.
    q = embs[i] / max(float(np.linalg.norm(embs[i])), 1e-12)
    x = embs[pool]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sims = x @ q
    cutoff = np.quantile(sims, 0.25)
    distant = pool[sims <= cutoff]
    sim_by_idx = {int(j): float(v) for j, v in zip(pool, sims)}
    target = lengths[i][source]

    def key(j):
        n = lengths[int(j)][source]
        # Prefer donors long enough to truncate exactly, then closest length.
        return (n < target, abs(n - target), sim_by_idx[int(j)], int(j))

    ranked = sorted((int(j) for j in distant), key=key)
    if len(ranked) < num_negatives:
        ranked = sorted((int(j) for j in pool), key=key)
    return ranked[:num_negatives]


def evaluate(rows, field):
    y = np.asarray([r["true_label"] for r in rows], dtype=int)
    scores = np.asarray([[r[field][s] for s in SOURCE_TYPES] for r in rows])
    pred = scores.argmax(1)
    per_type = {
        s: float(np.mean(pred[y == k] == k))
        for k, s in enumerate(SOURCE_TYPES)
    }
    return {
        "macro": float(np.mean(list(per_type.values()))),
        "per_type": per_type,
        "prediction_counts": {
            s: int(np.sum(pred == k)) for k, s in enumerate(SOURCE_TYPES)
        },
        "predictions": pred,
        "labels": y,
    }


def batch_score_sources(query, contexts, tok, llm, batch_size=5):
    """Vectorized equivalent of script_12.score_source for one query.

    Loss is still reduced separately for every sequence, so batching does not
    change the statistic.  Keeping modest micro-batches avoids materializing an
    unnecessarily large [batch, sequence, vocabulary] logits tensor.
    """
    scores = []
    instruction = (
        "Answer the following question using only the provided context. "
        "Be concise.\n\nContext:\n"
    )
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start:start + batch_size]
        prefixes = [instruction + c + "\n\nQuestion: " for c in chunk]
        full = [p + query + "\n\nAnswer:" for p in prefixes]
        enc_prefix = tok(prefixes, padding=True, return_tensors="pt",
                         truncation=True, max_length=900)
        enc_full = tok(full, padding=True, return_tensors="pt",
                       truncation=True, max_length=1024).to(S12.DEVICE)
        prefix_lens = enc_prefix["attention_mask"].sum(1).tolist()
        labels = enc_full["input_ids"].clone()
        labels[enc_full["attention_mask"] == 0] = -100
        for row, length in enumerate(prefix_lens):
            labels[row, :int(length)] = -100
        with torch.no_grad():
            logits = llm(**enc_full).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="none", ignore_index=-100,
        ).view(shift_labels.shape)
        valid = shift_labels.ne(-100)
        seq_loss = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
        scores.extend((-seq_loss).detach().cpu().tolist())
        del logits, shift_logits, shift_labels, token_loss
    return scores


def batch_score_question_only(query, contexts, tok, llm, batch_size=5,
                              use_chat_template=False):
    """Score only tokens overlapping the question's character span.

    The legacy scorer also includes the constant ``Answer:`` suffix in its
    loss.  Offset-based masking removes that prompt-token contamination and is
    robust to tokenizer merges at the prefix/question boundary.
    """
    scores = []
    instruction = (
        "Answer the following question using only the provided context. "
        "Be concise.\n\nContext:\n"
    )
    for start in range(0, len(contexts), batch_size):
        rendered, spans = [], []
        for context in contexts[start:start + batch_size]:
            if use_chat_template:
                marker = "\n\nQuestion: "
                user = f"Context:\n{context}{marker}{query}\n\nAnswer using only the context."
                text = tok.apply_chat_template(
                    [{"role": "system", "content": "You are a concise question-answering assistant."},
                     {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True,
                )
                q_start = text.rfind(marker + query)
                if q_start < 0:
                    raise RuntimeError("Could not locate question in rendered chat prompt")
                q_start += len(marker)
            else:
                prefix = instruction + context + "\n\nQuestion: "
                text = prefix + query
                q_start = len(prefix)
            rendered.append(text)
            spans.append((q_start, q_start + len(query)))

        enc = tok(rendered, padding=True, truncation=True, max_length=1024,
                  return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        labels = enc["input_ids"].clone()
        labels[:] = -100
        for row, (q_start, q_end) in enumerate(spans):
            for col, (a, b) in enumerate(offsets[row].tolist()):
                if b > q_start and a < q_end:
                    labels[row, col] = enc["input_ids"][row, col]
            if int(labels[row].ne(-100).sum()) == 0:
                raise RuntimeError("Question was truncated or produced no scored tokens")
        enc = enc.to(S12.DEVICE)
        labels = labels.to(S12.DEVICE)
        with torch.no_grad():
            logits = llm(**enc).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="none", ignore_index=-100,
        ).view(shift_labels.shape)
        valid = shift_labels.ne(-100)
        seq_loss = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
        scores.extend((-seq_loss).detach().cpu().tolist())
        del logits, shift_logits, shift_labels, token_loss
    return scores


def bootstrap_delta(base, alternative, reps=5000, seed=SEED):
    rng = np.random.RandomState(seed)
    y = base["labels"]
    a, b = base["predictions"], alternative["predictions"]
    values = []
    for _ in range(reps):
        ix = rng.randint(0, len(y), len(y))
        deltas = []
        for cls in range(3):
            mask = y[ix] == cls
            if not np.any(mask):
                break
            deltas.append(np.mean(b[ix][mask] == cls) - np.mean(a[ix][mask] == cls))
        if len(deltas) == 3:
            values.append(np.mean(deltas))
    values = np.asarray(values)
    return {
        "delta": float(alternative["macro"] - base["macro"]),
        "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "p_delta_le_zero": float(np.mean(values <= 0)),
    }


def serializable_eval(result):
    return {k: v for k, v in result.items() if k not in ("predictions", "labels")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--num-negatives", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--all", dest="all_data", action="store_true",
                    help="Use the complete test set instead of a balanced pilot")
    ap.add_argument("--split", choices=("test", "dev"), default="test")
    ap.add_argument("--score-positive", action="store_true",
                    help="Score positive contexts in this run rather than reuse script_39")
    ap.add_argument("--reader", choices=tuple(READERS), default="llama",
                    help="Locally cached causal reader (results use separate caches)")
    ap.add_argument("--scoring", choices=("legacy", "question_only", "chat_question_only"),
                    default="legacy")
    ap.add_argument("--context-variant",
                    choices=("raw", "records", "fact_reverse", "broken_association"),
                    default="raw")
    args = ap.parse_args()
    if args.num_negatives < 2:
        raise ValueError("Use at least two negatives to estimate a null distribution")

    data_file = S12.TEST_FILE if args.split == "test" else "mmrag_dev.json"
    emb_file = S12.EMB_CACHE if args.split == "test" else "query_emb_cache/dev_embs.npy"
    with open(data_file) as f:
        data = json.load(f)
    selected = list(range(len(data))) if args.all_data else S39.balanced_indices(
        data, args.per_class
    )
    data_tag = ("all" if args.split == "test" else "dev") if args.all_data else (
        f"pc{args.per_class}_{args.split}"
    )
    reader_name, reader_suffix = READERS[args.reader]
    tag_parts = [data_tag]
    if reader_suffix:
        tag_parts.append(reader_suffix)
    if args.scoring != "legacy":
        tag_parts.append("qonly" if args.scoring == "question_only" else "chatqonly")
    if args.context_variant != "raw":
        tag_parts.append(args.context_variant)
    tag = "_".join(tag_parts)
    if args.reader != "llama" and not args.score_positive:
        raise ValueError("Alternate readers require --score-positive; Llama raw scores cannot be reused")
    base = {}
    if not args.score_positive:
        base_path = OUT_DIR / f"ceg_cache_pc{args.per_class}.json"
        if not base_path.exists():
            raise FileNotFoundError(f"Run script_39 first: missing {base_path}")
        base = json.load(open(base_path))
        missing = [i for i in selected if str(i) not in base]
        if missing:
            raise RuntimeError(f"Base cache is missing {len(missing)} selected queries")

    embs = np.load(emb_file).astype(np.float32)
    if embs.shape[0] != len(data):
        raise RuntimeError(f"Embedding rows {embs.shape[0]} != data rows {len(data)}")
    # Retrieval is reader-independent and is intentionally shared across reader
    # replications; token lengths, donors, and likelihoods remain reader-specific.
    context_suffix = (data_tag if args.context_variant == "raw" else
                      f"{data_tag}_{args.context_variant}")
    context_cache_path = OUT_DIR / f"retrieved_contexts_{context_suffix}.json"
    if context_cache_path.exists():
        saved_contexts = json.load(open(context_cache_path))
        contexts = {int(i): v for i, v in saved_contexts.items()}
        if any(i not in contexts for i in selected):
            raise RuntimeError(f"Incomplete context cache: {context_cache_path}")
        print(f"Loaded {len(contexts)} cached retrieved contexts")
    elif args.context_variant != "raw":
        raise FileNotFoundError(
            f"Generate audited intervention contexts first: {context_cache_path}"
        )
    else:
        print("Loading FAISS indices and authoritative chunk-id maps...")
        indices = S12.load_indices()
        for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
            S12.build_pos_cache(ds)
        contexts = {}
        for i in tqdm(selected, desc="retrieve"):
            contexts[i] = {
                s: S12.retrieve_context(embs[i], s, indices, S12.TOP_K)
                for s in SOURCE_TYPES
            }
        with open(context_cache_path, "w") as f:
            json.dump(contexts, f)
        print(f"Saved retrieved contexts to {context_cache_path}")

    tok, llm = S39.load_llm_4bit(reader_name)
    lengths = {
        i: {s: len(S39.token_ids(contexts[i][s], tok)) for s in SOURCE_TYPES}
        for i in selected
    }
    donors = {
        i: {
            s: distant_donors(i, s, selected, embs, lengths, args.num_negatives)
            for s in SOURCE_TYPES
        }
        for i in selected
    }

    cache_path = OUT_DIR / f"null_evidence_cache_{tag}_m{args.num_negatives}.json"
    cache = json.load(open(cache_path)) if args.resume and cache_path.exists() else {}
    if cache:
        print(f"Loaded {len(cache)} cached queries")

    t0 = time.time()
    for pos, i in enumerate(tqdm(selected, desc="null score"), 1):
        key = str(i)
        if key in cache:
            continue
        negative_scores, donor_ids, pad_flags = {}, {}, {}
        all_negative_contexts = []
        score_slots = []
        for s in SOURCE_TYPES:
            target = lengths[i][s]
            source_donors = donors[i][s]
            donor_ids[s] = source_donors
            negative_scores[s] = []
            pad_flags[s] = []
            for m, j in enumerate(source_donors):
                fillers = [contexts[k][s] for k in source_donors[m + 1:]]
                fillers += [contexts[k][s] for k in source_donors[:m]]
                neg_ctx, padded = S39.exact_token_length(
                    contexts[j][s], target, tok, fillers=fillers
                )
                all_negative_contexts.append(neg_ctx)
                score_slots.append((s, m))
                negative_scores[s].append(None)
                pad_flags[s].append(bool(padded))

        positive_contexts = [contexts[i][s] for s in SOURCE_TYPES]
        score_contexts = (positive_contexts + all_negative_contexts
                          if args.score_positive else all_negative_contexts)
        if args.scoring == "legacy":
            batched_scores = batch_score_sources(
                data[i]["query"], score_contexts, tok, llm,
                batch_size=args.batch_size,
            )
        else:
            batched_scores = batch_score_question_only(
                data[i]["query"], score_contexts, tok, llm,
                batch_size=args.batch_size,
                use_chat_template=args.scoring == "chat_question_only",
            )
        if args.score_positive:
            raw = {s: float(batched_scores[k]) for k, s in enumerate(SOURCE_TYPES)}
            batched_scores = batched_scores[len(SOURCE_TYPES):]
        else:
            raw = {s: float(base[key]["raw"][s]) for s in SOURCE_TYPES}
        for (s, m), score in zip(score_slots, batched_scores):
            negative_scores[s][m] = float(score)

        null_mean = {s: float(np.mean(negative_scores[s])) for s in SOURCE_TYPES}
        gain = {s: raw[s] - null_mean[s] for s in SOURCE_TYPES}
        cache[key] = {
            "query_idx": int(i),
            "true_label": int(S39.hard_label(data[i])),
            "raw": raw,
            "negative_scores": negative_scores,
            "null_mean": null_mean,
            "gain": gain,
            "donor_query_idx": donor_ids,
            "negative_required_padding": pad_flags,
            "context_tokens": lengths[i],
        }
        if pos % 5 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            elapsed = time.time() - t0
            completed = sum(str(k) in cache for k in selected)
            rate = elapsed / max(1, completed)
            print(f"  [{completed}/{len(selected)}] approximate ETA="
                  f"{rate * (len(selected) - completed) / 60:.1f} min")

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    rows = [cache[str(i)] for i in selected]
    # Pooled within-query null dispersion preserves the per-query null mean
    # while providing a much more stable scale than an SD from only M samples.
    pooled_sd = {}
    for s in SOURCE_TYPES:
        residuals = []
        for r in rows:
            values = np.asarray(r["negative_scores"][s])
            residuals.extend((values - values.mean()).tolist())
        pooled_sd[s] = max(float(np.std(residuals, ddof=1)), 1e-6)
    for r in rows:
        r["null_z"] = {s: r["gain"][s] / pooled_sd[s] for s in SOURCE_TYPES}

    raw_eval = evaluate(rows, "raw")
    gain_eval = evaluate(rows, "gain")
    z_eval = evaluate(rows, "null_z")
    summary = {
        "method": "multi-negative null-normalized evidence",
        "n": len(rows),
        "per_class": args.per_class,
        "num_negatives": args.num_negatives,
        "reader": reader_name,
        "reader_role": ("primary" if args.reader == "llama" else
                        "different-family, math-tuned stress test"),
        "scoring": args.scoring,
        "context_variant": args.context_variant,
        "seed": SEED,
        "donor_selection": (
            "least-similar query-embedding quartile; length matched; "
            "no routing-label or gold-answer use"
        ),
        "raw": serializable_eval(raw_eval),
        "mean_gain": serializable_eval(gain_eval),
        "null_z": serializable_eval(z_eval),
        "gain_vs_raw": bootstrap_delta(raw_eval, gain_eval),
        "null_z_vs_raw": bootstrap_delta(raw_eval, z_eval),
        "pooled_within_query_null_sd": pooled_sd,
        "negative_padding_rate": {
            s: float(np.mean([
                flag for r in rows for flag in r["negative_required_padding"][s]
            ]))
            for s in SOURCE_TYPES
        },
    }
    out = OUT_DIR / f"null_evidence_summary_{tag}_m{args.num_negatives}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 72)
    for name, result in [("raw", raw_eval), ("gain", gain_eval), ("null_z", z_eval)]:
        print(f"{name:8s} macro={result['macro']:.4f}  {result['per_type']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
