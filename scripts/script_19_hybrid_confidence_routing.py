"""
script_19_hybrid_confidence_routing.py — Confidence-tiered hybrid routing (offline, zero API cost)

IDEA (adapted from ConflictRAG's two-stage confidence tiering, arXiv:2605.17301)
  Route to a single source when the router's softmax confidence is high;
  fall back to union (no_routing) retrieval when it is low. Under GPT-4o-mini,
  pure routing scores F1 0.437 (phase3) vs pure union 0.442. If router errors
  concentrate in low-confidence queries, a hybrid can beat both.

WHY ZERO COST
  Per-query F1/EM for BOTH arms are already cached for all 1,286 test queries:
    phase5_results/reader_scaling_openai_predictions_phase3_seed42.jsonl
    phase5_results/reader_scaling_openai_predictions_phase4_seed42.jsonl
    phase5_results/reader_scaling_openai_predictions_no_routing.jsonl
  The hybrid is a per-query SELECTION between cached answers — no generation.

HONESTY GUARDRAILS (do not weaken these)
  Sweeping tau on the test set and reporting the best tau is selection bias.
  We therefore report three numbers per configuration:
    1. full sweep curve (analysis / figure only, NOT a headline claim)
    2. a-priori tau = 0.7 (ConflictRAG's fixed threshold, chosen before looking)
    3. 2-fold cross-fitted estimate: split test in half by parity of index,
       choose tau* on fold A, evaluate on fold B, and vice versa; report the
       pooled F1. This is the only number eligible for the paper's tables.

USAGE
  python script_19_hybrid_confidence_routing.py

OUTPUT
  phase5_results/hybrid_confidence_summary.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from sourceformer import SourceFormerK3

RESULTS_DIR = Path("phase5_results")
CKPT = Path("checkpoints/sourceformer_k3_seed42_best.pt")
EMB = Path("query_emb_cache/test_embs.npy")
TEST_FILE = "mmrag_test.json"

TAUS = np.round(np.arange(0.34, 1.001, 0.01), 3)
APRIORI_TAU = 0.7  # fixed before any sweep; matches ConflictRAG's threshold


def load_preds(name):
    path = RESULTS_DIR / f"reader_scaling_openai_predictions_{name}.jsonl"
    return [json.loads(l) for l in open(path)]


def hybrid_f1(routed_f1, union_f1, conf, tau):
    take_routed = conf >= tau
    f1 = np.where(take_routed, routed_f1, union_f1)
    return float(f1.mean()), float(take_routed.mean())


def crossfit(routed_f1, union_f1, conf):
    """2-fold cross-fitting: pick tau on one fold, evaluate on the other."""
    n = len(conf)
    idx = np.arange(n)
    foldA, foldB = idx[idx % 2 == 0], idx[idx % 2 == 1]
    out_f1 = np.empty(n)
    chosen = {}
    for fit, evl, tag in [(foldA, foldB, "A->B"), (foldB, foldA, "B->A")]:
        best_tau, best = None, -1.0
        for tau in TAUS:
            f1, _ = hybrid_f1(routed_f1[fit], union_f1[fit], conf[fit], tau)
            if f1 > best:
                best, best_tau = f1, tau
        take = conf[evl] >= best_tau
        out_f1[evl] = np.where(take, routed_f1[evl], union_f1[evl])
        chosen[tag] = float(best_tau)
    return float(out_f1.mean()), chosen


def main():
    # --- cached per-query results, alignment asserted on id ---
    union = load_preds("no_routing")
    union_f1 = np.array([r["f1"] for r in union])
    test = json.load(open(TEST_FILE))
    assert all(t["id"] == r["id"] for t, r in zip(test, union)), "order mismatch"

    # --- router confidence from checkpoint + cached embeddings (local, no LLM) ---
    embs = np.load(EMB)
    assert embs.shape[0] == len(test)
    model = SourceFormerK3()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(embs).float()), dim=-1).numpy()
    conf = probs.max(axis=1)
    print(f"Router confidence: min={conf.min():.3f} median={np.median(conf):.3f} "
          f"max={conf.max():.3f}")

    summary = {"n": len(test), "union_f1": float(union_f1.mean()),
               "apriori_tau": APRIORI_TAU, "configs": {}}

    for name in ["phase3_seed42", "phase4_seed42"]:
        routed = load_preds(name)
        assert all(t["id"] == r["id"] for t, r in zip(test, routed)), "order mismatch"
        routed_f1 = np.array([r["f1"] for r in routed])

        sweep = []
        for tau in TAUS:
            f1, frac = hybrid_f1(routed_f1, union_f1, conf, tau)
            sweep.append({"tau": float(tau), "f1": f1, "frac_routed": frac})
        best = max(sweep, key=lambda s: s["f1"])

        ap_f1, ap_frac = hybrid_f1(routed_f1, union_f1, conf, APRIORI_TAU)
        cf_f1, cf_taus = crossfit(routed_f1, union_f1, conf)

        cfg = {
            "pure_routed_f1": float(routed_f1.mean()),
            "pure_union_f1": float(union_f1.mean()),
            "sweep_best": best,                      # analysis only, biased
            "apriori_tau0.7": {"f1": ap_f1, "frac_routed": ap_frac},
            "crossfit_f1": cf_f1,                    # unbiased, paper-eligible
            "crossfit_taus": cf_taus,
            "sweep": sweep,
        }
        summary["configs"][name] = cfg

        print(f"\n=== {name} ===")
        print(f"  pure routed F1     : {routed_f1.mean():.4f}")
        print(f"  pure union  F1     : {union_f1.mean():.4f}")
        print(f"  hybrid tau=0.7     : {ap_f1:.4f}  (routed on {ap_frac:.1%} of queries)")
        print(f"  hybrid cross-fitted: {cf_f1:.4f}  (taus {cf_taus})")
        print(f"  sweep best (biased): {best['f1']:.4f} at tau={best['tau']} "
              f"(routed {best['frac_routed']:.1%})")

    out = RESULTS_DIR / "hybrid_confidence_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
