"""Counterfactual Evidence Gain pilot for heterogeneous-source routing.

For each query q and source s, score:

    raw_s = log p(q | retrieved context_s)
    neg_s = log p(q | length-matched context from the same source,
                  retrieved for a different query)
    CEG_s = raw_s - neg_s

The same-source negative estimates the source-format fluency prior while
preserving serialization and context length.  If CEG improves routing over raw
likelihood, the result supports a format-prior mechanism rather than the
invalidated short-context-collapse account.

The script also performs an exploratory evidence-preserving KG sentenceifying
intervention.  Its result is diagnostic, not by itself a causal estimate,
because exact sentenceification can change tokenization even after matching the
final token budget.

Usage (always in the chestx environment for GPU runs):
  python script_39_counterfactual_evidence_gain.py --per-class 2   # smoke
  python script_39_counterfactual_evidence_gain.py --per-class 100 # pilot

Outputs:
  phase9_results/ceg_cache_pc{N}.json
  phase9_results/ceg_summary_pc{N}.json
"""

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import script_12_prefrag_conf as S12


OUT_DIR = Path("phase9_results")
OUT_DIR.mkdir(exist_ok=True)
SOURCE_TYPES = S12.SOURCE_TYPES_K3
SOURCE_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}
SEED = 42
BOOTSTRAP_REPS = 5000


def type_scores(item):
    s = item["dataset_score"]
    return np.asarray([
        s.get("nq", 0) + s.get("triviaqa", 0),
        s.get("ott", 0) + s.get("tat", 0),
        s.get("kg", 0),
    ], dtype=float)


def hard_label(item):
    return int(type_scores(item).argmax())


def acceptable_labels(item):
    v = type_scores(item)
    return np.flatnonzero(np.isclose(v, v.max())).tolist()


def balanced_indices(data, per_class, seed=SEED):
    rng = np.random.RandomState(seed)
    selected = []
    for label in range(3):
        candidates = np.asarray([i for i, x in enumerate(data)
                                 if hard_label(x) == label], dtype=int)
        if len(candidates) < per_class:
            raise ValueError(f"class {label} has only {len(candidates)} records")
        selected.extend(rng.choice(candidates, per_class, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def load_llm_4bit(model_name=None):
    """Load a locally cached causal reader with the benchmark quantization.

    ``model_name`` is optional so the original Llama experiment remains exactly
    backward compatible while allowing a separately cached reader-family check.
    """
    model_name = model_name or S12.LLM_NAME
    print(f"Loading {model_name} (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    # The benchmark host keeps the model locally and may not have outbound DNS.
    # local_files_only also prevents a tokenizer metadata request from making an
    # otherwise reproducible cached run depend on network availability.
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = {
        "quantization_config": bnb,
        "device_map": "auto",
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    gpu_limit = os.environ.get("CEG_GPU_MAX_GIB")
    if gpu_limit:
        load_kwargs.update({
            "max_memory": {0: f"{gpu_limit}GiB", "cpu": "22GiB"},
            "offload_folder": "/tmp/ceg_model_offload",
            "offload_state_dict": True,
        })
        print(f"  CUDA allocation capped at {gpu_limit} GiB; overflow offloads to CPU")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return tok, model


def token_ids(text, tok):
    return tok(text, add_special_tokens=False)["input_ids"]


def exact_token_length(text, target, tok, fillers=()):
    """Return text with exactly target tokenizer tokens.

    When the base negative is too short, append other non-repeating same-source
    donor contexts before truncating.  A period token is used only as a final
    fallback, and the caller records how often padding was needed.
    """
    ids = list(token_ids(text, tok))
    used_padding = len(ids) < target
    for filler in fillers:
        if len(ids) >= target:
            break
        ids.extend(token_ids("\n\n" + filler, tok))
    if not ids:
        ids = token_ids("No relevant context.", tok)
    if len(ids) < target:
        period = token_ids(".", tok) or [tok.eos_token_id]
        while len(ids) < target:
            ids.extend(period)
    return tok.decode(ids[:target], skip_special_tokens=True), used_padding


def answer_strings(item):
    a = item.get("answer", "")
    if isinstance(a, str):
        return [a] if a.strip() else []
    if isinstance(a, list):
        return [str(x) for x in a if str(x).strip()]
    if isinstance(a, dict):
        out = []
        for v in a.values():
            if isinstance(v, list):
                out.extend(str(x) for x in v)
            elif v is not None:
                out.append(str(v))
        return [x for x in out if x.strip()]
    return [str(a)] if a is not None else []


def contains_gold(context, item):
    low = context.lower()
    return any(len(a.strip()) >= 3 and a.strip().lower() in low
               for a in answer_strings(item))


TRIPLE_RE = re.compile(r"^\s*(.*?)\s+--\s+(.*?)\s+-->\s+(.*?)\s*$")


def sentenceify_kg(context):
    """Convert parseable `subject -- relation --> object` lines to prose."""
    out, parsed = [], 0
    for line in context.splitlines():
        m = TRIPLE_RE.match(line)
        if m:
            subj, rel, obj = (x.strip() for x in m.groups())
            out.append(f"{subj} has {rel}: {obj}.")
            parsed += 1
        elif line.strip():
            out.append(line.strip())
    return "\n".join(out), parsed


def choose_donor(i, source, contexts, lengths, data, selected):
    """Closest-length same-source donor, excluding self and gold leakage."""
    target_len = lengths[i][source]
    candidates = []
    for j in selected:
        if j == i or hard_label(data[j]) == hard_label(data[i]):
            continue
        ctx = contexts[j][source]
        if not ctx or contains_gold(ctx, data[i]):
            continue
        candidates.append((abs(lengths[j][source] - target_len), j))
    if not candidates:
        for j in selected:
            if j != i and contexts[j][source] and not contains_gold(contexts[j][source], data[i]):
                candidates.append((abs(lengths[j][source] - target_len), j))
    if not candidates:
        raise RuntimeError(f"no eligible donor for query {i}, source {source}")
    candidates.sort()
    return candidates[0][1]


def evaluate(score_rows, data_by_idx, field):
    preds, labels, set_hits = [], [], []
    for r in score_rows:
        scores = r[field]
        pred = int(np.argmax([scores[s] for s in SOURCE_TYPES]))
        y = hard_label(data_by_idx[r["query_idx"]])
        preds.append(pred); labels.append(y)
        set_hits.append(pred in acceptable_labels(data_by_idx[r["query_idx"]]))
    preds, labels = np.asarray(preds), np.asarray(labels)
    per_type = {}
    for j, s in enumerate(SOURCE_TYPES):
        mask = labels == j
        per_type[s] = float(np.mean(preds[mask] == j))
    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro": float(np.mean(list(per_type.values()))),
        "per_type": per_type,
        "set_valued_accuracy": float(np.mean(set_hits)),
        "prediction_counts": {s: int(np.sum(preds == j))
                              for j, s in enumerate(SOURCE_TYPES)},
        "preds": preds,
        "labels": labels,
    }


def bootstrap_macro_delta(raw_eval, ceg_eval, reps=BOOTSTRAP_REPS, seed=SEED):
    rng = np.random.RandomState(seed)
    raw_p, ceg_p, y = raw_eval["preds"], ceg_eval["preds"], raw_eval["labels"]
    n = len(y)
    vals = []
    for _ in range(reps):
        ix = rng.randint(0, n, n)
        per_raw, per_ceg = [], []
        for cls in range(3):
            m = y[ix] == cls
            if not np.any(m):
                break
            per_raw.append(np.mean(raw_p[ix][m] == cls))
            per_ceg.append(np.mean(ceg_p[ix][m] == cls))
        if len(per_raw) == 3:
            vals.append(np.mean(per_ceg) - np.mean(per_raw))
    vals = np.asarray(vals)
    return {
        "delta": float(ceg_eval["macro"] - raw_eval["macro"]),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "p_delta_le_zero": float(np.mean(vals <= 0)),
        "bootstrap_reps": int(len(vals)),
    }


def strip_arrays(d):
    return {k: v for k, v in d.items() if k not in ("preds", "labels")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(S12.TEST_FILE) as f:
        data = json.load(f)
    selected = balanced_indices(data, args.per_class)
    selected_set = set(selected)
    print(f"Balanced pilot: {len(selected)} queries ({args.per_class}/class), seed={SEED}")

    embs = np.load(S12.EMB_CACHE).astype(np.float32)
    if embs.shape[0] != len(data):
        raise RuntimeError(f"embedding cache has {embs.shape[0]} rows, expected {len(data)}")

    print("Loading FAISS indices and authoritative chunk-id maps...")
    indices = S12.load_indices()
    for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
        S12.build_pos_cache(ds)

    print("Retrieving positive contexts for pilot and donor pool...")
    contexts = {}
    for i in tqdm(selected, desc="retrieve"):
        contexts[i] = {s: S12.retrieve_context(embs[i], s, indices, S12.TOP_K)
                       for s in SOURCE_TYPES}

    tok, llm = load_llm_4bit()
    lengths = {i: {s: len(token_ids(contexts[i][s], tok)) for s in SOURCE_TYPES}
               for i in selected}

    donors = {i: {s: choose_donor(i, s, contexts, lengths, data, selected)
                  for s in SOURCE_TYPES} for i in selected}

    cache_path = OUT_DIR / f"ceg_cache_pc{args.per_class}.json"
    cache = {}
    if args.resume and cache_path.exists():
        cache = json.load(open(cache_path))
        print(f"Loaded {len(cache)} cached queries")

    t0 = time.time()
    for pos, i in enumerate(tqdm(selected, desc="CEG score"), 1):
        key = str(i)
        if key in cache:
            continue
        raw, neg, ceg = {}, {}, {}
        donor_ids, pad_flags = {}, {}
        for s in SOURCE_TYPES:
            pos_ctx = contexts[i][s]
            target = lengths[i][s]
            j = donors[i][s]
            donor_ids[s] = int(j)
            filler_ids = [k for k in selected
                          if k not in (i, j) and hard_label(data[k]) != hard_label(data[i])]
            neg_ctx, padded = exact_token_length(
                contexts[j][s], target, tok,
                fillers=[contexts[k][s] for k in filler_ids[:4]],
            )
            pad_flags[s] = bool(padded)
            raw[s] = float(S12.score_source(data[i]["query"], pos_ctx, tok, llm))
            neg[s] = float(S12.score_source(data[i]["query"], neg_ctx, tok, llm))
            ceg[s] = raw[s] - neg[s]

        kg_sent, parsed_pos = sentenceify_kg(contexts[i]["kg"])
        kg_neg_base, parsed_neg = sentenceify_kg(contexts[donors[i]["kg"]]["kg"])
        target = lengths[i]["kg"]
        kg_sent, sent_pad = exact_token_length(kg_sent, target, tok)
        kg_neg_sent, neg_sent_pad = exact_token_length(kg_neg_base, target, tok)
        sent_pos_score = float(S12.score_source(data[i]["query"], kg_sent, tok, llm))
        sent_neg_score = float(S12.score_source(data[i]["query"], kg_neg_sent, tok, llm))

        cache[key] = {
            "query_idx": int(i),
            "true_label": hard_label(data[i]),
            "raw": raw,
            "negative": neg,
            "ceg": ceg,
            "context_tokens": lengths[i],
            "donor_query_idx": donor_ids,
            "negative_required_padding": pad_flags,
            "kg_sentenceified": {
                "positive_score": sent_pos_score,
                "negative_score": sent_neg_score,
                "ceg": sent_pos_score - sent_neg_score,
                "parsed_positive_triples": parsed_pos,
                "parsed_negative_triples": parsed_neg,
                "positive_required_padding": sent_pad,
                "negative_required_padding": neg_sent_pad,
            },
        }
        if pos % 10 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            elapsed = time.time() - t0
            eta = elapsed / max(1, pos) * (len(selected) - pos)
            print(f"  [{pos}/{len(selected)}] ETA={eta/60:.1f} min")

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    rows = [cache[str(i)] for i in selected]
    raw_eval = evaluate(rows, data, "raw")
    ceg_eval = evaluate(rows, data, "ceg")
    delta = bootstrap_macro_delta(raw_eval, ceg_eval)

    kg_raw_ceg = np.asarray([r["ceg"]["kg"] for r in rows])
    kg_sent_ceg = np.asarray([r["kg_sentenceified"]["ceg"] for r in rows])
    summary = {
        "method": "Counterfactual Evidence Gain pilot",
        "n": len(rows),
        "per_class": args.per_class,
        "seed": SEED,
        "reader": S12.LLM_NAME,
        "quantization": "4-bit NF4",
        "top_k_per_constituent_index": S12.TOP_K,
        "max_context_chars": S12.MAX_CTX_CHARS,
        "raw": strip_arrays(raw_eval),
        "ceg": strip_arrays(ceg_eval),
        "macro_delta_bootstrap": delta,
        "mean_score_by_source": {
            "raw": {s: float(np.mean([r["raw"][s] for r in rows])) for s in SOURCE_TYPES},
            "negative": {s: float(np.mean([r["negative"][s] for r in rows])) for s in SOURCE_TYPES},
            "ceg": {s: float(np.mean([r["ceg"][s] for r in rows])) for s in SOURCE_TYPES},
        },
        "kg_sentenceification": {
            "mean_raw_ceg": float(kg_raw_ceg.mean()),
            "mean_sentenceified_ceg": float(kg_sent_ceg.mean()),
            "mean_delta": float((kg_sent_ceg - kg_raw_ceg).mean()),
            "median_delta": float(np.median(kg_sent_ceg - kg_raw_ceg)),
            "parsed_positive_triples_total": int(sum(
                r["kg_sentenceified"]["parsed_positive_triples"] for r in rows
            )),
        },
        "negative_padding_rate": {s: float(np.mean([
            r["negative_required_padding"][s] for r in rows
        ])) for s in SOURCE_TYPES},
    }
    out = OUT_DIR / f"ceg_summary_pc{args.per_class}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Raw macro: {raw_eval['macro']:.4f}  {raw_eval['per_type']}")
    print(f"CEG macro: {ceg_eval['macro']:.4f}  {ceg_eval['per_type']}")
    print(f"Delta: {delta['delta']:+.4f}, 95% bootstrap CI={delta['ci95']}")
    print(f"KG sentenceification mean CEG delta: "
          f"{summary['kg_sentenceification']['mean_delta']:+.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
