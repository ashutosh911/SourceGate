"""
script_33_e5_rescore_corrected.py  —  E5-base end-to-end F1/EM under the
                                      CORRECTED reader pipeline

WHY
  Section 6.10 reports a retrieval-encoder robustness check (BGE-base vs
  E5-base-v2). Its recall@picked figures are comparable across encoders, but
  its F1/EM were produced by script_17_encoder_robustness.py, which generates
  through script_7_evaluation.generate_answer + phase4_components.LLMConfig.
  That is the PRE-FIX reader configuration, and it differs from the corrected
  one (script_10_reader_scaling.LocalLlamaReader) in two ways that both
  depress token-F1:

    1. SYSTEM PROMPT
       pre-fix   "Answer the question using only the provided context.
                  Be concise."
       corrected "Answer the question based on the provided context. Be
                  concise. Respond with ONLY the answer in a few words, no
                  explanation. If unsure, give your best guess."
       The pre-fix prompt permits prose answers, which score poorly as
       token-F1 against short gold spans.

    2. TRUNCATION
       pre-fix   left-truncates the TOKENISED prompt when it exceeds
                 max_seq_len, which removes <|begin_of_text|> and the system
                 header on long contexts, corrupting the chat template.
       corrected truncates the CONTEXT to 3000 characters before templating,
                 leaving the template intact.

  Because the difference is in generation, the existing
  encoder_robustness_e5_cache.json cannot simply be re-scored: those strings
  were produced by the wrong prompt. This script regenerates E5's answers
  with the corrected reader and scores them with the same f1/em functions
  used everywhere else.

  BGE-base numbers are NOT recomputed -- they already come from the corrected
  pipeline (script_10). Only the E5 side moves, which is the whole point:
  afterwards the two encoders are finally like-for-like.

RETRIEVAL
  Canonical convention, matching script_32 / Table 4: a source type's
  constituent indices are searched, merged by score, and the global top-10
  kept. no_routing pools all five E5 indices and keeps the global top-10.

USAGE
  conda activate chestx && python script_33_e5_rescore_corrected.py [--limit N]

OUTPUT
  phase5_results/encoder_robustness_e5_corrected.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import faiss
import torch

import script_10_reader_scaling as S10          # corrected reader + metrics
from phase4_components import ChunkDB, format_chunks

RESULTS_DIR = Path("phase5_results")
E5_DIR = Path("faiss_indices_e5")
E5_MODEL = "intfloat/e5-base-v2"
E5_QUERY_PREFIX = "query: "
SOURCE_TYPES = ["text", "table", "kg"]
TYPE_TO_DATASETS = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
DATASETS = ["nq", "triviaqa", "ott", "tat", "kg"]
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def e5_pool(h, mask):
    """E5 requires attention-masked MEAN pooling (not BGE's CLS token)."""
    masked = h.masked_fill(~mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)


def encode_queries_e5(queries, batch=64):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(E5_MODEL)
    mdl = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(queries), batch):
            enc = tok([E5_QUERY_PREFIX + q for q in queries[i:i + batch]],
                      padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(DEVICE)
            h = mdl(**enc).last_hidden_state
            e = e5_pool(h, enc["attention_mask"])
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N queries (smoke test)")
    args = ap.parse_args()

    records = S10.load_test_records("mmrag_test.json")
    if args.limit:
        records = records[: args.limit]
    n = len(records)
    print(f"{n} test queries")

    cache_p = RESULTS_DIR / "e5_query_embs_test.npy"
    if cache_p.exists() and np.load(cache_p).shape[0] == n:
        embs = np.load(cache_p)
        print("  E5 query embeddings: cache hit")
    else:
        print("  Encoding test queries with E5 (mean pooling)...")
        embs = encode_queries_e5([r["query"] for r in records])
        if not args.limit:
            np.save(cache_p, embs)

    print("Loading E5 indices...")
    idx, cids = {}, {}
    for ds in DATASETS:
        idx[ds] = faiss.read_index(str(E5_DIR / ds / "index.faiss"))
        cids[ds] = list(np.load(E5_DIR / ds / "chunk_ids.npy", allow_pickle=True))
        print(f"  {ds}: {idx[ds].ntotal} vectors")

    def merged_top_k(q, datasets):
        """Search each index, merge by score, keep the global top-10."""
        pool = []
        for ds in datasets:
            sc, pos = idx[ds].search(q, min(TOP_K, idx[ds].ntotal))
            for j, p in enumerate(pos[0]):
                if 0 <= p < len(cids[ds]):
                    pool.append((float(sc[0][j]), cids[ds][p]))
        pool.sort(key=lambda t: -t[0])
        return [c for _, c in pool[:TOP_K]]

    chunk_db = ChunkDB()
    reader = S10.LocalLlamaReader()          # corrected prompt + truncation

    results, preds_dump = {}, {}
    for method in ("oracle", "no_routing"):
        print(f"\n=== {method} (E5 retrieval, corrected reader) ===")
        f1s, ems, hits, rows = [], [], [], []
        t0 = time.time()
        for i, r in enumerate(records):
            q = embs[i: i + 1]
            if method == "oracle":
                ds_list = TYPE_TO_DATASETS[SOURCE_TYPES[r["oracle_label"]]]
            else:
                ds_list = DATASETS
            picked = merged_top_k(q, ds_list)

            d = chunk_db.get_many(picked)
            context = format_chunks([d[c] for c in picked if c in d])
            pred = reader.generate(r["query"], context)

            f1 = S10.f1_score(pred, r["answer"])
            em = S10.exact_match(pred, r["answer"])
            f1s.append(f1); ems.append(em)
            if method == "oracle":
                rel = {c for c, s in r["relevant_chunks"].items() if s > 0}
                hits.append(float(bool(set(picked) & rel)))
            rows.append({"id": r["id"], "gold": r["answer"], "pred": pred,
                         "f1": f1, "em": em})
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{n}  F1={np.mean(f1s):.4f}  "
                      f"({(i+1)/max(el,1):.2f} q/s)")

        results[method] = {
            "method": method, "encoder": "E5-base-v2", "n": n,
            "reader": "Llama-3.1-8B-Instruct-4bit (corrected pipeline)",
            "f1_mean": float(np.mean(f1s)), "em_mean": float(np.mean(ems)),
            "recall_at_picked": float(np.mean(hits)) if hits else None,
        }
        preds_dump[method] = rows
        print(f"  {method}: F1={np.mean(f1s):.4f}  EM={np.mean(ems):.4f}"
              + (f"  R@picked={np.mean(hits):.4f}" if hits else ""))

    # pre-fix values, for the record
    prefix_p = RESULTS_DIR / "encoder_robustness_e5_metrics.json"
    if prefix_p.exists():
        old = json.load(open(prefix_p))
        for m in results:
            if m in old:
                results[m]["f1_prefix_pipeline"] = old[m]["f1_mean"]
                results[m]["em_prefix_pipeline"] = old[m]["em_mean"]

    out = {"note": ("E5-base end-to-end scores regenerated under the corrected "
                    "reader pipeline (script_10 SYSTEM_PROMPT + context[:3000] "
                    "truncation). Directly comparable to the BGE-base rows of "
                    "Table 4/9, which already use that pipeline."),
           "bge_base_reference": {"oracle": {"f1": 0.364, "em": 0.261,
                                             "recall_at_picked": 0.802},
                                  "no_routing": {"f1": 0.364, "em": 0.262}},
           "e5_corrected": results}
    json.dump(out, open(RESULTS_DIR / "encoder_robustness_e5_corrected.json", "w"),
              indent=2)
    json.dump(preds_dump,
              open(RESULTS_DIR / "encoder_robustness_e5_corrected_preds.json", "w"),
              indent=2)

    print("\n" + "=" * 68)
    print(f"{'method':<12}{'F1 (pre-fix)':>14}{'F1 (corrected)':>16}{'BGE-base':>11}")
    print("=" * 68)
    for m, v in results.items():
        ref = out["bge_base_reference"][m]["f1"]
        print(f"{m:<12}{v.get('f1_prefix_pipeline', float('nan')):>14.4f}"
              f"{v['f1_mean']:>16.4f}{ref:>11.3f}")
    print("\nSaved → phase5_results/encoder_robustness_e5_corrected.json")


if __name__ == "__main__":
    main()
