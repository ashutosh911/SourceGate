"""
script_20_equal_length_control.py  —  Equal-length context control (Reviewer 2)

PURPOSE
  R2's central objection: "length is never separated from the other properties
  of the sources … a direct experiment would settle it: pad or truncate the
  contexts so that every source has the same length, then score them again.
  This requires no retraining."

  This script re-scores PrefRAG-Conf under THREE equalised-length conditions:
    A) ALL SOURCES TRUNCATED TO A SHORT 8-TOKEN BUDGET
       Stress-tests format when every source is equally short.
       If collapse persists → format (not length) drives the signal.
    B) ALL SOURCES PADDED/TRUNCATED TO A COMMON 85-TOKEN BUDGET
       Middle ground: tests whether collapse is specific to extreme length ratios.
    C) ALL SOURCES PADDED/TRUNCATED TO A LONG 254-TOKEN BUDGET
       Stress-tests what happens when short contexts are expanded.

  For PADDING we repeat the context text cyclically to reach the target length.
  This is admittedly artificial, but it provides a clean length-equalization signal
  and is exactly what R2 asked for. We report the result honestly.

COST
  1286 queries × 3 sources × 3 conditions = 11,574 LLM forward passes
  Each pass: single-source, top-10 chunks, no generation (NLL scoring only)
  At ~2s/query (NF4 Llama 3.1 8B): ~6-7 hours on RTX 5070 Ti
  With caching of conditions: resumable, will reuse any prior passes.

USAGE
  python script_20_equal_length_control.py                    # all conditions
  python script_20_equal_length_control.py --condition A      # KG-length only (fastest)
  python script_20_equal_length_control.py --n 100            # smoke test

OUTPUT
  phase5_results/equal_length_question_v2_summary.json
  phase5_results/equal_length_question_v2_cache_{condition}.json
"""

import argparse, json, gc, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from script_12_prefrag_conf import (
    load_indices, build_pos_cache, retrieve_context, load_or_encode_test,
    score_source,
    SOURCE_TYPES_K3, SOURCE_IDX, LLM_NAME, TOP_K,
    RESULTS_DIR, TEST_FILE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── length targets (token counts) ────────────────────────────────────────────
LENGTH_CONDITIONS = {
    "A": {"name": "short_budget", "target_tokens": 8,   "label": "All sources @ 8-token budget"},
    "B": {"name": "common_budget", "target_tokens": 85,  "label": "All sources @ 85-token budget"},
    "C": {"name": "long_budget", "target_tokens": 254, "label": "All sources @ 254-token budget"},
}

# ── load frozen LLM for NLL scoring ─────────────────────────────────────────
def load_llm():
    print(f"Loading {LLM_NAME} (4-bit NF4)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return model, tok

# ── truncate / pad context to target token length ────────────────────────────
def equalize_context(context: str, target_tokens: int, tokenizer) -> str:
    """Truncate or cyclically pad context to exactly target_tokens tokens."""
    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        # Empty context: create minimal padding
        ids = tokenizer("no context available", add_special_tokens=False)["input_ids"]
    if len(ids) >= target_tokens:
        # Truncate
        ids = ids[:target_tokens]
    else:
        # Cyclically repeat to reach target length
        reps = (target_tokens // len(ids)) + 1
        ids = (ids * reps)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

# ── per-query source scoring ─────────────────────────────────────────────────
def score_all_sources(query, answer, context_strings, model, tokenizer):
    """Returns source -> mean question-token log-probability.

    The answer argument is retained for call-site compatibility but is not
    scored. Delegating to the canonical PrefRAG-Conf scorer ensures that the
    equal-length intervention changes only the context.
    """
    scores = {}
    for src, ctx in context_strings.items():
        scores[src] = score_source(query, ctx, tokenizer, model)
    return scores

# ── routing evaluation ───────────────────────────────────────────────────────
def evaluate_routing(scores_list, test_data):
    """Compute macro accuracy from list of {source -> logprob} dicts."""
    from collections import defaultdict as dd
    per_type = dd(lambda: [0, 0])
    for i, scores in enumerate(scores_list):
        if scores is None:
            continue
        pred = max(scores, key=scores.get)
        pred_idx = SOURCE_IDX[pred]
        # Get true label from dataset score argmax
        ds = test_data[i].get("dataset_score", {})
        # Match the paper's K=3 label construction and script_12 exactly:
        # constituent-dataset relevance is summed within each source type.
        by_type = {"text": ds.get("nq", 0) + ds.get("triviaqa", 0),
                   "table": ds.get("ott", 0) + ds.get("tat", 0),
                   "kg": ds.get("kg", 0)}
        true_src = max(by_type, key=by_type.get)
        per_type[true_src][1] += 1
        if pred == true_src:
            per_type[true_src][0] += 1
    per_type_acc = {s: per_type[s][0]/per_type[s][1]
                    for s in SOURCE_TYPES_K3 if per_type[s][1] > 0}
    macro = float(np.mean(list(per_type_acc.values())))
    return macro, per_type_acc

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["A", "B", "C"],
                    default=None, help="Run one condition only")
    ap.add_argument("--n", type=int, default=None, help="Limit to first N queries")
    args = ap.parse_args()

    conditions = ([args.condition] if args.condition
                  else list(LENGTH_CONDITIONS.keys()))

    print("=" * 70)
    print("Equal-length context control (Reviewer 2 mechanism isolation)")
    print("=" * 70)
    for c in conditions:
        print(f"  Condition {c}: {LENGTH_CONDITIONS[c]['label']}")
    print()

    with open(TEST_FILE) as f:
        test_data = json.load(f)
    n = len(test_data) if args.n is None else min(args.n, len(test_data))
    test_data = test_data[:n]

    print(f"Loading FAISS indices and building position caches...")
    faiss_indices = load_indices()
    for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
        build_pos_cache(ds)

    print("Encoding test queries (reuses cache if available)...")
    embs = load_or_encode_test([{"query": t["query"]} for t in test_data])

    print("Loading frozen LLM...")
    model, tokenizer = load_llm()

    # Also load original (variable-length) scores for comparison
    orig_path = RESULTS_DIR / "prefrag_conf_results.json"
    if orig_path.exists():
        orig_scores = {r["query_idx"]: r["source_scores"]
                       for r in json.load(open(orig_path))}
    else:
        orig_scores = {}

    summary = {"n": n, "conditions": {}}

    for cond_key in conditions:
        cond = LENGTH_CONDITIONS[cond_key]
        target = cond["target_tokens"]
        cache_path = RESULTS_DIR / f"equal_length_question_v2_cache_{cond_key}.json"

        print(f"\n{'='*70}")
        print(f"Condition {cond_key}: {cond['label']}")
        print(f"{'='*70}")

        # Load cache
        cache = {}
        if cache_path.exists():
            cache = json.load(open(cache_path))
            print(f"  Loaded {len(cache)} cached entries from {cache_path}")

        scores_list = [None] * n
        t0 = time.time()

        for i in tqdm(range(n), desc=f"Cond {cond_key}"):
            key = str(i)
            if key in cache:
                scores_list[i] = cache[key]
                continue

            item = test_data[i]
            q_emb = embs[i:i+1]

            # Retrieve and equalize each source
            equalized_contexts = {}
            for src in SOURCE_TYPES_K3:
                raw_ctx = retrieve_context(q_emb[0], src, faiss_indices, TOP_K)
                equalized_contexts[src] = equalize_context(
                    raw_ctx, target, tokenizer)

            scores = score_all_sources(
                item["query"], item.get("answer", ""),
                equalized_contexts, model, tokenizer)
            scores_list[i] = scores
            cache[key] = scores

            if (i + 1) % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  [{i+1}/{n}] ETA {eta/60:.0f}min")

        with open(cache_path, "w") as f:
            json.dump(cache, f)

        macro, per_type_acc = evaluate_routing(scores_list, test_data)

        # Correlation: equalized context length vs NLL
        # (all contexts now have same token count, so if correlation persists
        #  it cannot be length-driven)
        equalized_nlls = defaultdict(list)
        for i, sc in enumerate(scores_list):
            if sc:
                for src, logprob in sc.items():
                    if not np.isnan(logprob):
                        equalized_nlls[src].append(-logprob)

        cond_result = {
            "target_tokens": target,
            "macro": macro,
            "per_type": per_type_acc,
            "mean_nll_per_source": {s: float(np.mean(v))
                                    for s, v in equalized_nlls.items()},
        }

        # Compare to original (variable-length)
        if orig_scores:
            orig_macro, _ = evaluate_routing(
                [orig_scores.get(i) for i in range(n)], test_data)
            cond_result["original_macro"] = orig_macro
            cond_result["delta_macro"] = macro - orig_macro

        summary["conditions"][cond_key] = cond_result

        print(f"\n  Results for Condition {cond_key} ({cond['label']}):")
        print(f"    Macro accuracy:   {macro:.3f}")
        print(f"    Per-type:         {per_type_acc}")
        print(f"    Mean NLL/source:  {cond_result['mean_nll_per_source']}")
        if "original_macro" in cond_result:
            print(f"    Original macro:   {cond_result['original_macro']:.3f}")
            print(f"    Delta:            {cond_result['delta_macro']:+.3f}")
        print()
        print("  INTERPRETATION:")
        if macro > 0.55:
            print("  -> Equalizing length RECOVERS routing signal.")
            print("     Length was the primary confound, not source format.")
            print("     This STRENGTHENS the paper's mechanism claim.")
        elif macro < 0.40:
            print("  -> Collapse PERSISTS at equal length.")
            print("     Source format (not length alone) drives the collapse.")
            print("     The paper should say 'format confound' not 'length confound'.")
            print("     Both the KG-prose mismatch and the length work together.")
        else:
            print("  -> Partial recovery (between random 0.337 and original 0.334).")
            print("     Both length and format contribute to the collapse.")

    out_path = RESULTS_DIR / "equal_length_question_v2_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {out_path}")

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Condition':<35} {'Macro':>8} {'Delta':>8}")
    print("-"*55)
    for cond_key, res in summary["conditions"].items():
        label = LENGTH_CONDITIONS[cond_key]["label"]
        delta = res.get("delta_macro", float("nan"))
        print(f"  {label:<33} {res['macro']:>8.3f} {delta:>8.3f}")
    print(f"  {'Original (variable length)':<33} {summary['conditions'][list(summary['conditions'].keys())[0]].get('original_macro', float('nan')):>8.3f}")
    print("-"*55)
    print(f"  Reference: random=0.337, logistic=0.642, SG Phase3=0.737")


if __name__ == "__main__":
    main()
