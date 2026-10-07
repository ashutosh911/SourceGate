"""
script_57_c1_nuisance_decomposition.py

C1 re-examined from a different angle.

C1 as published: generation-confidence routing collapses under a source-format
/ context-length confound.  The collapse does not exist in corrected data, so
the PHENOMENON is gone.  But the MECHANISM -- reader NLL carrying a
source-format nuisance term unrelated to evidence -- is separately testable.

script_40 (CEG) tested it by subtracting a raw donor score and failed
(contrastive_M3 0.5846 < raw 0.5896).  A raw donor is ONE noisy measurement:
subtracting it removes whatever nuisance exists and injects a sample's worth of
new noise.  The M1<M2<M3 ordering rising back toward raw is that noise
shrinking as donors are averaged -- it does not tell us whether a nuisance
term is there underneath.

This script separates the two by shrinkage.  Write the evidence-free level
measured on donors as  n[q,s]  (same source format, content from another
query).  Correct the real score by  lambda * n  and sweep lambda:

    corrected[q,s] = real[q,s] - lambda * n[q,s]

  lambda = 0  -> raw (0.5896)
  lambda = 1  -> full donor subtraction (= CEG M3, 0.5846)

If a real nuisance term exists but donor noise was masking it, macro peaks at
an INTERIOR lambda strictly above both endpoints.  If no nuisance exists,
macro falls monotonically from lambda=0 and the optimum is at the boundary.
The same sweep is run on the rank-2 source-constant part alone (which is
low-variance and cannot inject per-query noise) and on the interaction part.

CONVENTION: ceg_scores.npz stores NEGATIVE NLL -- higher is better, route by
argmax.  Verified against script_40's reported raw macro of 0.5896.

NOTE ON LENGTH: lens in ceg_scores.npz is identically 800 for every query and
every source, so the context-length half of C1 CANNOT be tested from that
file.  It is tested separately here against context_budget_test_3200.npz
(same 1286 queries, 4x budget).  context_budget_test_800.npz is only 40
queries and is the dataset-ordered biased slice from trap #1 -- not used.

CPU only.  Full 1286-query test set throughout.
"""
import argparse
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="phase5_results/ceg_scores.npz",
                help="CEG score file; use ceg_scores_d3.npz for the frozen "
                     "3-donor copy or ceg_scores.npz for the newest run")
ap.add_argument("--tag", default="",
                help="suffix for the output file, e.g. _d10")
args = ap.parse_args()

rng = np.random.default_rng(20260929)
SRC = ["text", "table", "kg"]
OUT = f"phase5_results/c1_nuisance_decomposition{args.tag}.json"
print(f"scores: {args.scores}  ->  {OUT}")

z = np.load(args.scores, allow_pickle=True)
real = z["real"]            # (Q,3) NEGATIVE NLL, higher = better
neg = z["neg"]              # (Q,3,D) donor (evidence-free) negative NLL
gold = z["gold"]
Q, S = real.shape
D = neg.shape[2]
print(f"Q={Q} S={S} donors={D}")
print("gold:", {s: int((gold == i).sum()) for i, s in enumerate(SRC)})


def macro(scores, g):
    pick = scores.argmax(1)
    return float(np.mean([(pick[g == c] == c).mean() for c in range(S)]))


def demean(x):
    """Remove per-query constant a_q; it cancels under argmax over s."""
    return x - x.mean(1, keepdims=True)


res = {}
raw = macro(real, gold)
res["raw_macro"] = raw
print(f"\nraw macro (argmax real)                  {raw:.4f}   [script_40 says 0.5896]")
assert abs(raw - 0.5896) < 1e-3, "convention mismatch vs script_40"

# ---------------------------------------------------------------- components
n_full = demean(neg.mean(2))          # evidence-free level, per (q,s)
b_s = n_full.mean(0, keepdims=True)   # source-constant part (low variance)
n_int = n_full - b_s                  # per-query interaction part (noisy)
real_dm = demean(real)

print("\nevidence-free source effect from donors (nats, demeaned):")
print("  " + "   ".join(f"{SRC[c]}={b_s[0,c]:+.4f}" for c in range(S)))
print("real source effect (format+evidence):")
print("  " + "   ".join(f"{SRC[c]}={real_dm[:,c].mean():+.4f}" for c in range(S)))
span_n = b_s.max() - b_s.min()
span_r = real_dm.mean(0).max() - real_dm.mean(0).min()
res["source_effect_donors"] = {SRC[c]: float(b_s[0, c]) for c in range(S)}
res["source_effect_real"] = {SRC[c]: float(real_dm[:, c].mean()) for c in range(S)}
res["source_span_donors"] = float(span_n)
res["source_span_real"] = float(span_r)
res["nuisance_share_of_source_gap"] = float(span_n / span_r)
print(f"  source span: donors {span_n:.4f} / real {span_r:.4f} "
      f"-> evidence-free part is {100*span_n/span_r:.1f}% of the real gap")

# ------------------------------------------------------------ lambda sweeps
lams = np.round(np.arange(-0.5, 2.01, 0.05), 3)
sweeps = {}
for name, comp in [("full_donor", n_full), ("source_const", b_s * np.ones((Q, 1))),
                   ("interaction", n_int)]:
    ms = [macro(real_dm - L * comp, gold) for L in lams]
    ms = np.array(ms)
    i = int(ms.argmax())
    sweeps[name] = {"lambda_grid": lams.tolist(), "macro": ms.tolist(),
                    "best_lambda": float(lams[i]), "best_macro": float(ms[i]),
                    "macro_at_0": float(ms[np.where(lams == 0)[0][0]]),
                    "macro_at_1": float(ms[np.where(lams == 1)[0][0]])}
    interior = 0.0 < lams[i] < 2.0 and ms[i] > max(sweeps[name]["macro_at_0"],
                                                   sweeps[name]["macro_at_1"]) + 1e-9
    sweeps[name]["interior_optimum"] = bool(interior)
    print(f"\n  sweep[{name:>13}]  best lambda={lams[i]:+.2f}  macro={ms[i]:.4f}  "
          f"(l=0: {sweeps[name]['macro_at_0']:.4f}, l=1: {sweeps[name]['macro_at_1']:.4f})"
          f"  interior_opt={interior}")
res["lambda_sweeps"] = sweeps

# ------------------------------------------ bootstrap the best correction
best_name = max(sweeps, key=lambda k: sweeps[k]["best_macro"])
best_lam = sweeps[best_name]["best_lambda"]
comp = {"full_donor": n_full, "source_const": b_s * np.ones((Q, 1)),
        "interaction": n_int}[best_name]
corrected = real_dm - best_lam * comp
diffs = np.empty(2000)
for i in range(2000):
    idx = rng.integers(0, Q, Q)
    diffs[i] = macro(corrected[idx], gold[idx]) - macro(real[idx], gold[idx])
res["best"] = {
    "component": best_name, "lambda": best_lam,
    "macro": sweeps[best_name]["best_macro"],
    "delta_vs_raw_mean": float(diffs.mean()),
    "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
}
print(f"\nBEST: {best_name} at lambda={best_lam:+.2f} -> macro "
      f"{sweeps[best_name]['best_macro']:.4f}")
print(f"  delta vs raw {diffs.mean():+.4f}  95% CI "
      f"[{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]  "
      f"p={res['best']['p_two_sided']:.4f}")
print("  NOTE: lambda was chosen on the same data it is scored on -- this is an")
print("        UPPER BOUND on what nuisance removal can buy, not a clean estimate.")

# ------------------------------------------------------ context-length axis
print("\n" + "=" * 64)
print("CONTEXT-LENGTH AXIS (800 vs 3200 char budget, same 1286 queries)")
b32 = np.load("phase5_results/context_budget_test_3200.npz", allow_pickle=True)
s32, g32 = b32["scores"], b32["gold"]
if s32.shape[0] == Q and np.array_equal(g32, gold):
    dm32 = demean(s32)
    m32 = macro(s32, gold)
    res["budget"] = {
        "macro_800": raw, "macro_3200": m32, "delta": float(m32 - raw),
        "source_effect_800": {SRC[c]: float(real_dm[:, c].mean()) for c in range(S)},
        "source_effect_3200": {SRC[c]: float(dm32[:, c].mean()) for c in range(S)},
    }
    print(f"  macro  800 chars: {raw:.4f}")
    print(f"  macro 3200 chars: {m32:.4f}   ({m32-raw:+.4f})")
    print("  source effect  800: " +
          "  ".join(f"{SRC[c]}={real_dm[:,c].mean():+.4f}" for c in range(S)))
    print("  source effect 3200: " +
          "  ".join(f"{SRC[c]}={dm32[:,c].mean():+.4f}" for c in range(S)))
    shift = {SRC[c]: float(dm32[:, c].mean() - real_dm[:, c].mean()) for c in range(S)}
    res["budget"]["source_effect_shift"] = shift
    print("  format x length interaction (shift): " +
          "  ".join(f"{k}={v:+.4f}" for k, v in shift.items()))
else:
    print("  SKIPPED: gold/shape mismatch vs ceg_scores")
    res["budget"] = {"skipped": True}

# --------------------------------------------- conditional-collapse check
print("\n" + "=" * 64)
print("CONDITIONAL COLLAPSE (does concentration appear in a subset?)")
hit = np.load("phase5_results/hit_matrix_test_strict.npy")
pick = real.argmax(1)
res["conditional"] = {}
for label, mask in [
    ("all", np.ones(Q, bool)),
    ("gold source retrieved", hit[np.arange(Q), gold] == 1),
    ("gold source MISSED", hit[np.arange(Q), gold] == 0),
    ("no source retrieved", hit.sum(1) == 0),
    ("all sources retrieved", hit.sum(1) == S),
]:
    if mask.sum() == 0:
        continue
    dist = np.array([(pick[mask] == c).mean() for c in range(S)])
    mac = float(np.mean([(pick[mask & (gold == c)] == c).mean()
                         for c in range(S) if (mask & (gold == c)).sum() > 0]))
    res["conditional"][label] = {"n": int(mask.sum()),
                                 "pick_dist": dist.tolist(), "macro": mac,
                                 "max_share": float(dist.max())}
    print(f"  {label:<24} n={mask.sum():>5}  picks "
          f"{'/'.join(f'{d:.3f}' for d in dist)}  max={dist.max():.3f}  macro={mac:.4f}")
print("  (C1-style collapse would need max share near 1.0 in some subset)")

res["reference"] = {"raw_published": 0.5883, "source_calibration_oracle": 0.652,
                    "bge_confidence": 0.661, "sourcegate": 0.737, "random": 0.337}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nwrote {OUT}")
