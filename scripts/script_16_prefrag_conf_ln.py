"""
script_16_prefrag_conf_ln.py — Length-normalized PrefRAG-Conf (Option A diagnostic)

CLAIM UNDER TEST
  PrefRAG-Conf routes to the source whose context maximizes the mean log-prob of
  the question tokens. In script_12 this collapsed to KG (macro 0.334; KG recall
  0.914, text 0.038, table 0.050). Our diagnosis: KG "triple" contexts are far
  SHORTER than text/table contexts, and short contexts perturb the LM's question
  distribution less, so they win the confidence contest regardless of relevance.
  This is a context-length artifact, not a routing signal.

WHY PLAIN LENGTH-NORMALIZATION IS A NO-OP HERE
  script_12's score is already out.loss = *mean* log-prob over the QUESTION tokens.
  The question is identical across the 3 candidate sources, so its token count is
  constant per query -> dividing by question length cannot change the argmax.
  The only length that varies across sources is the CONTEXT length. So the honest
  test of the length-bias hypothesis is to normalize / calibrate against context
  length. This script does that four ways:

    raw            argmax mean-logprob                     (reproduces the collapse)
    per_ctx_token  argmin  NLL / context_token_len         (literal "divide by length")
    len_residual   argmax  residual of logprob ~ ctx_len   (regress length out)
    zcal           argmax  per-source z(logprob)           (remove fixed per-source offset)

  If any rule recovers macro toward the learned router (~0.72), that's a cheap fix
  worth reporting. If none do, the collapse is not mere length and the LEARNED
  router is justified — either outcome strengthens the paper.

INPUTS (no LLM forward pass needed — reuses saved scores)
  phase5_results/prefrag_conf_results.json   per-query source_scores (mean logprob)
  FAISS + chunk DB + Llama tokenizer          to recover per-(query,source) ctx length

USAGE
  python script_16_prefrag_conf_ln.py                 # uses saved scores
  python script_16_prefrag_conf_ln.py --n 100         # smoke test
  python script_16_prefrag_conf_ln.py --recompute_len # rebuild the length cache

OUTPUT
  phase5_results/prefrag_conf_ln_summary.json
  phase5_results/prefrag_conf_ctx_lengths.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# Reuse script_12's retrieval so contexts are reconstructed identically.
from script_12_prefrag_conf import (
    load_indices, build_pos_cache, retrieve_context, load_or_encode_test,
    eval_routing, SOURCE_TYPES_K3, SOURCE_IDX, LLM_NAME, TOP_K,
    RESULTS_DIR, TEST_FILE,
)

RAW_SCORES = RESULTS_DIR / "prefrag_conf_results.json"
LEN_CACHE = RESULTS_DIR / "prefrag_conf_ctx_lengths.npy"


def load_context_token_lengths(test_data, top_k, recompute=False):
    """Per-(query, source) context token length under the Llama tokenizer.

    Reconstructs the exact context string script_12 fed to the scorer
    (retrieve_context -> capped at MAX_CTX_CHARS) and counts its tokens.
    No LLM forward pass — just retrieval + tokenizer.
    """
    n = len(test_data)
    if LEN_CACHE.exists() and not recompute:
        arr = np.load(LEN_CACHE)
        if arr.shape[0] >= n:
            print(f"  ctx-length cache hit: {LEN_CACHE} shape={arr.shape}")
            return arr[:n]
        print("  ctx-length cache too small — recomputing.")

    print("  Recomputing per-(query,source) context token lengths...")
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    embs = load_or_encode_test([{"query": t["query"]} for t in test_data])
    faiss_indices = load_indices()
    for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
        build_pos_cache(ds)

    lens = np.zeros((n, len(SOURCE_TYPES_K3)), dtype=np.int32)
    for i in range(n):
        for j, src in enumerate(SOURCE_TYPES_K3):
            ctx = retrieve_context(embs[i], src, faiss_indices, top_k)
            lens[i, j] = len(tok(ctx, add_special_tokens=False)["input_ids"])
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n}")
    np.save(LEN_CACHE, lens)
    print(f"  Saved {LEN_CACHE}")
    return lens


def report(name, preds, labels, kg_idx):
    acc, macro, per_type = eval_routing(preds, labels)
    kg_rate = float((preds == kg_idx).mean())
    print(f"  {name:<14} acc={acc:.3f} macro={macro:.3f}  "
          f"text={per_type.get('text', float('nan')):.3f} "
          f"table={per_type.get('table', float('nan')):.3f} "
          f"kg={per_type.get('kg', float('nan')):.3f}   KG-picked={kg_rate:.1%}")
    return {"acc": acc, "macro": macro, "per_type": per_type, "kg_pick_rate": kg_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--recompute_len", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("PrefRAG-Conf-LN — is the KG collapse a context-length artifact?")
    print("=" * 74)

    if not RAW_SCORES.exists():
        raise SystemExit(f"Missing {RAW_SCORES}. Run script_12_prefrag_conf.py first.")
    with open(RAW_SCORES) as f:
        raw = json.load(f)
    raw.sort(key=lambda r: r["query_idx"])
    if args.n:
        raw = raw[:args.n]

    with open(TEST_FILE) as f:
        test_data = json.load(f)
    test_data = [test_data[r["query_idx"]] for r in raw]  # align to scored queries

    # logprob[i, j] = mean question-token log-prob under source j (higher = better)
    logprob = np.array([[r["source_scores"][s] for s in SOURCE_TYPES_K3] for r in raw],
                       dtype=np.float64)
    nll = -logprob
    labels = np.array([r["true_label"] for r in raw], dtype=np.int64)
    ctx_len = load_context_token_lengths(test_data, args.top_k,
                                         recompute=args.recompute_len).astype(np.float64)
    ctx_len = np.maximum(ctx_len, 1.0)
    kg_idx = SOURCE_IDX["kg"]

    # ---- Diagnostic: the mechanism ------------------------------------------
    print("\nMechanism (per-source means across queries):")
    print(f"  {'source':<8}{'ctx_tokens':>12}{'mean_NLL':>12}")
    for j, s in enumerate(SOURCE_TYPES_K3):
        print(f"  {s:<8}{ctx_len[:, j].mean():>12.1f}{nll[:, j].mean():>12.3f}")
    corr = np.corrcoef(ctx_len.ravel(), nll.ravel())[0, 1]
    print(f"  corr(ctx_tokens, NLL) over all (query,source) pairs = {corr:+.3f}")
    print("  (negative corr => longer context -> higher NLL -> penalized => "
          "short KG wins on raw confidence)")

    # ---- Routing rules -------------------------------------------------------
    print("\nRouting rules:")
    results = {}

    # raw: highest mean log-prob (= lowest NLL). Reproduces script_12.
    results["raw"] = report("raw", logprob.argmax(1), labels, kg_idx)

    # per_ctx_token: literal "divide token-level NLL by context length", pick min.
    results["per_ctx_token"] = report("per_ctx_token",
                                      (nll / ctx_len).argmin(1), labels, kg_idx)

    # len_residual: regress logprob on ctx_len (pooled), route by max residual.
    x = ctx_len.ravel()
    y = logprob.ravel()
    b1, b0 = np.polyfit(x, y, 1)
    residual = logprob - (b0 + b1 * ctx_len)
    results["len_residual"] = report("len_residual", residual.argmax(1), labels, kg_idx)

    # zcal: per-source z-score of logprob (removes fixed per-source offset).
    mu = logprob.mean(0, keepdims=True)
    sd = logprob.std(0, keepdims=True) + 1e-9
    results["zcal"] = report("zcal", ((logprob - mu) / sd).argmax(1), labels, kg_idx)

    print("\nReference macro (K=3 test):  learned SF Phase3=0.737  MLP-HardCE=0.702  "
          "BGE-conf=0.661  random=0.337")

    summary = {
        "method": "PrefRAG-Conf-LN",
        "n_queries": len(raw),
        "top_k": args.top_k,
        "mechanism": {
            "per_source_mean_ctx_tokens": {
                s: float(ctx_len[:, j].mean()) for j, s in enumerate(SOURCE_TYPES_K3)},
            "per_source_mean_nll": {
                s: float(nll[:, j].mean()) for j, s in enumerate(SOURCE_TYPES_K3)},
            "corr_ctxlen_nll": float(corr),
        },
        "rules": results,
    }
    out = RESULTS_DIR / "prefrag_conf_ln_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out}")

    best = max(results, key=lambda k: results[k]["macro"])
    print(f"\nBest rule: {best} (macro={results[best]['macro']:.3f}). "
          + ("Length correction RECOVERS routing — report as a cheap fix."
             if results[best]["macro"] > 0.55 and best != "raw"
             else "No length correction recovers macro — the collapse is not mere "
                  "length; the LEARNED router is justified."))


if __name__ == "__main__":
    main()
