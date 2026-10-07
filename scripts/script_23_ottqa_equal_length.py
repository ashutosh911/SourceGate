"""
script_23_ottqa_equal_length.py  —  OTT-QA equal-length context control

PURPOSE
  R2 (and the audit) ask: run the same equal-length control on OTT-QA.
  Two possible outcomes:
    A) Table NLL advantage PERSISTS at equal length
       -> Format drives OTT-QA collapse (same conclusion as mmRAG)
       -> Single consistent rule: format-mismatched source gets lower NLL
    B) Table NLL advantage DISAPPEARS at equal length
       -> Length drives OTT-QA, format drives mmRAG
       -> Different mechanisms on two benchmarks; weaker general claim

  ottqa_k2_test.json has pre-retrieved table_context and passage_context
  for each query -- no FAISS needed. We just truncate/pad to equal tokens
  and re-score with PrefRAG-Conf (question-token NLL under frozen LLM).

TWO CONDITIONS:
  A) Both sources truncated to passage median (85 tokens) -- same as mmRAG
  B) Both sources truncated to table median (235 tokens)

  Condition A is the most informative: it removes the table length advantage.
  If table NLL advantage persists -> format (not length) drives OTT-QA collapse.

RUNTIME
  2214 queries × 2 sources × 2 conditions = ~8,856 forward passes.
  No generation; NLL scoring only (single forward pass per query-source pair).
  ~1-2 hours on RTX 5070 Ti.

USAGE
  python script_23_ottqa_equal_length.py
  python script_23_ottqa_equal_length.py --condition A   # passage median only (~45 min)
  python script_23_ottqa_equal_length.py --n 100         # smoke test

OUTPUT
  phase8_results_ottqa/ottqa_equal_length_summary.json
  phase8_results_ottqa/ottqa_equal_length_cache_{A,B}.json  (resumable)
"""

import argparse, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE  = "phase8_results_ottqa/ottqa_k2_test.json"
RESULTS_DIR = Path("phase8_results_ottqa")

# ── length targets (from prefrag_conf_k2_summary.json) ───────────────────────
LENGTH_CONDITIONS = {
    "A": {"target": 85,  "label": "Both @ passage median (85 tok)"},
    "B": {"target": 235, "label": "Both @ table median (235 tok)"},
}

SOURCES = ["table", "passage"]   # K=2

# ── load frozen LLM ──────────────────────────────────────────────────────────
def load_llm():
    print(f"Loading {LLM_MODEL} (4-bit NF4)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, quantization_config=bnb, device_map="auto")
    model.eval()
    return model, tok

# ── equalize context to target token length ───────────────────────────────────
def equalize(text: str, target: int, tokenizer) -> str:
    """Truncate or cyclically pad text to exactly `target` tokens."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer("no context", add_special_tokens=False)["input_ids"]
    if len(ids) >= target:
        ids = ids[:target]
    else:
        reps = (target // len(ids)) + 1
        ids = (ids * reps)[:target]
    return tokenizer.decode(ids, skip_special_tokens=True)

# ── NLL scoring (question tokens, same as PrefRAG-Conf) ──────────────────────
PROMPT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
          "Answer using only the context.<|eot_id|>\n"
          "<|start_header_id|>user<|end_header_id|>\n"
          "Context: {context}\nQuestion: {query}<|eot_id|>\n"
          "<|start_header_id|>assistant<|end_header_id|>\n")

@torch.no_grad()
def score_question_nll(model, tokenizer, query: str, context: str) -> float:
    """Mean log-prob of question tokens conditioned on context (PrefRAG-Conf style)."""
    ctx_prompt = PROMPT.format(context=context, query="")
    full_prompt = PROMPT.format(context=context, query=query)

    ctx_ids  = tokenizer(ctx_prompt,  return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(DEVICE)
    full_ids = tokenizer(full_prompt, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(DEVICE)

    if full_ids.shape[1] > 1500:
        return float("nan")

    labels = full_ids.clone()
    labels[:, :ctx_ids.shape[1]] = -100   # mask context, score only question

    out = model(input_ids=full_ids, labels=labels)
    return -out.loss.item()               # return log-prob (higher = more confident)

# ── routing evaluation ────────────────────────────────────────────────────────
def evaluate_routing(scores_list, test_data):
    """scores_list: list of dicts {source -> logprob}. Returns macro, per_type."""
    per_type = defaultdict(lambda: [0, 0])   # correct, total
    for i, scores in enumerate(scores_list):
        if scores is None:
            continue
        pred  = max(scores, key=scores.get)
        label = test_data[i]["label"]        # "table" or "passage"
        per_type[label][1] += 1
        if pred == label:
            per_type[label][0] += 1
    per_type_acc = {s: per_type[s][0] / per_type[s][1]
                    for s in SOURCES if per_type[s][1] > 0}
    macro = float(np.mean(list(per_type_acc.values())))
    return macro, per_type_acc

# ── original (variable-length) routing for reference ─────────────────────────
def original_routing(test_data):
    """Reproduces Table 11's PrefRAG-Conf raw = 0.389."""
    # From prefrag_conf_k2_summary.json: raw macro = 0.389,
    # table pick-rate 59.8%. We derive it from the NLL means:
    # table NLL mean = 2.868, passage NLL mean = 3.011
    # -> table wins most queries (lower NLL = higher confidence)
    # We don't have per-query scores saved, so use the summary stats
    print("  (Original variable-length: macro=0.389 from prefrag_conf_k2_summary.json)")
    return 0.389, {"table": 0.428, "passage": 0.350}

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["A", "B"], default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    conditions = [args.condition] if args.condition else ["A", "B"]

    print("=" * 70)
    print("OTT-QA equal-length context control")
    print("=" * 70)
    for c in conditions:
        print(f"  Condition {c}: {LENGTH_CONDITIONS[c]['label']}")
    print()

    with open(DATA_FILE) as f:
        test_data = json.load(f)
    n = len(test_data) if args.n is None else min(args.n, len(test_data))
    test_data = test_data[:n]
    print(f"  {n} test queries  |  device: {DEVICE}")

    model, tokenizer = load_llm()

    # Reference: original variable-length routing
    orig_macro, orig_per_type = original_routing(test_data)

    summary = {
        "n": n,
        "original": {
            "macro": orig_macro,
            "per_type": orig_per_type,
            "mean_nll": {
                "table":   -2.868,   # from prefrag_conf_k2_summary.json
                "passage": -3.011,
            },
            "ctx_tokens": {
                "table":   235.1,
                "passage":  84.7,
            },
        },
        "conditions": {},
    }

    for cond_key in conditions:
        cond   = LENGTH_CONDITIONS[cond_key]
        target = cond["target"]
        cache_path = RESULTS_DIR / f"ottqa_equal_length_cache_{cond_key}.json"

        print(f"\n{'='*70}")
        print(f"Condition {cond_key}: {cond['label']}")
        print(f"{'='*70}")

        cache = {}
        if cache_path.exists():
            raw_cache = json.load(open(cache_path))
            # Validate: skip entries where ALL scores are nan (from prior CUDA crash)
            for k, v in raw_cache.items():
                if any(s is not None and not (isinstance(s, float) and s != s)
                       for s in v.values()):
                    cache[k] = v
            print(f"  Loaded {len(cache)}/{len(raw_cache)} valid cached entries "
                  f"({len(raw_cache)-len(cache)} nan entries discarded)")

        scores_list = [None] * n
        nlls = defaultdict(list)
        t0 = time.time()

        for i in tqdm(range(n), desc=f"OTT-QA cond {cond_key}"):
            key = str(i)
            if key in cache:
                scores_list[i] = cache[key]
                for src, v in cache[key].items():
                    nlls[src].append(-v)
                continue

            item = test_data[i]
            item_scores = {}
            for src in SOURCES:
                ctx_raw  = item[f"{src}_context"]
                ctx_eq   = equalize(ctx_raw, target, tokenizer)
                try:
                    logprob = score_question_nll(model, tokenizer, item["query"], ctx_eq)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        torch.cuda.empty_cache()
                        logprob = float("nan")
                    else:
                        raise
                item_scores[src] = logprob
                if not np.isnan(logprob):
                    nlls[src].append(-logprob)

            scores_list[i] = item_scores
            cache[key] = item_scores

            if (i + 1) % 100 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                torch.cuda.empty_cache()   # prevent VRAM fragmentation
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  [{i+1}/{n}] ETA {eta/60:.0f}min  "
                      f"VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB")

        with open(cache_path, "w") as f:
            json.dump(cache, f)

        macro, per_type = evaluate_routing(scores_list, test_data)
        mean_nll = {s: float(np.mean(v)) for s, v in nlls.items()}

        cond_result = {
            "target_tokens": target,
            "macro":         macro,
            "per_type":      per_type,
            "mean_nll":      mean_nll,
            "delta_macro":   macro - orig_macro,
        }
        summary["conditions"][cond_key] = cond_result

        print(f"\n  Results — Condition {cond_key}: {cond['label']}")
        print(f"    Macro:        {macro:.3f}  (original: {orig_macro:.3f}, delta: {macro-orig_macro:+.3f})")
        tbl = per_type.get('table', float('nan'))
        pas = per_type.get('passage', float('nan'))
        print(f"    Per-type:     table={tbl:.3f}  passage={pas:.3f}")
        print(f"    Mean NLL:     table={mean_nll.get('table','N/A'):.3f}  "
              f"passage={mean_nll.get('passage','N/A'):.3f}")
        nll_gap = mean_nll.get("passage", 0) - mean_nll.get("table", 0)
        print(f"    NLL gap (passage - table): {nll_gap:+.3f}  "
              f"(original: {-3.011 - (-2.868):+.3f})")
        print()
        print("  INTERPRETATION:")
        if nll_gap > 0.05:
            print("  -> Table still has lower NLL at equal length.")
            print("     FORMAT (not length alone) drives the OTT-QA collapse.")
            print("     Both benchmarks share the same mechanism: format-mismatched")
            print("     source gets lower NLL regardless of context length.")
        elif abs(nll_gap) <= 0.05:
            print("  -> NLL gap eliminated at equal length.")
            print("     LENGTH drives the OTT-QA collapse.")
            print("     mmRAG (format) and OTT-QA (length) have DIFFERENT mechanisms.")
        else:
            print("  -> Passage has lower NLL at equal length (table advantage reversed).")
            print("     Equal-length over-corrects; result inconclusive.")

    # Save
    out = RESULTS_DIR / "ottqa_equal_length_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Condition':<40} {'Macro':>8} {'Delta':>8} {'NLL gap':>10}")
    print("  " + "-"*68)
    print(f"  {'Original (variable length)':<40} {orig_macro:>8.3f} {'':>8} "
          f"  {-3.011-(-2.868):>8.3f}")
    for ck, res in summary["conditions"].items():
        nll_gap = res["mean_nll"].get("passage",0) - res["mean_nll"].get("table",0)
        print(f"  {LENGTH_CONDITIONS[ck]['label']:<40} {res['macro']:>8.3f} "
              f"{res['delta_macro']:>+8.3f}   {nll_gap:>8.3f}")
    print()
    print(f"  Reference: random=0.484, SourceGate K=2=0.676")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
