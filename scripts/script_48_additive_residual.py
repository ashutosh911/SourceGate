"""
script_48_additive_residual.py  --  can the complementary information in reader
                                    likelihood actually be cashed in?

THE SITUATION
  script_44 established, under two independent protocols that agree, that
  reader likelihood carries source information SourceGate's logits do not:

      dev-fit / test-frozen     +0.0292  CI [+0.0034, +0.0546]  p=0.027
      cross-fit within test     +0.0429  CI [+0.0194, +0.0662]  p=0.0006

  and that this is NOT merely retrieval-success information, since substituting
  a predicted-retrieval-hit feature reproduces none of it.

  But neither combiner beat the PLAIN SourceGate argmax (0.7327 and 0.7268 vs
  0.7415).  The reason is architectural, not informational: refitting a
  logistic regression on three logits with ~800 fitting examples destroys more
  of SourceGate than likelihood puts back.  The comparison was rigged against
  the combination by construction.

THE FIX
  Do not refit SourceGate.  Add to it.

      score_s(q) = log_softmax(SG(q))_s + lambda * z_s(q)

  where z_s is the reader's likelihood standardised per source using DEV
  statistics only.  At lambda = 0 this is SourceGate exactly, so the family
  contains the incumbent and a single scalar decides how much of the reader's
  opinion to admit.  One parameter cannot overfit 766 dev queries, and the
  comparison is no longer rigged: the combination can only lose if the reader's
  information is genuinely unusable.

  A per-source variant (three lambdas) is reported alongside to show whether
  the useful part is concentrated in one source type -- the parallel Qwen work
  found table/KG AUC gains and a text non-effect, which predicts it might be.

PROTOCOLS  (same two as script_44; believe nothing that fails both)
  P1  lambda chosen on dev, applied once to frozen test.
      Caveat: SourceGate's checkpoint was early-stopped on dev macro, so dev is
      not perfectly clean with respect to its logits.
  P2  cross-fit within test: choose lambda on half A, apply to half B, and
      vice versa.  Clean with respect to SourceGate, which never saw test.

PREDICTIONS, FIXED BEFORE RUNNING
  A1  The information is cashable.  -> lambda* != 0 on dev, and test macro
      exceeds 0.7415 with a bootstrap CI excluding zero.
  A2  It is real but not cashable.  -> lambda* ~ 0, or test macro rises by less
      than noise.  The honest claim then is informational, not performance:
      the reader knows something the router does not, and it is too small and
      too noisy to exploit.

USAGE
  conda activate chestx && python script_48_additive_residual.py

OUTPUT
  phase5_results/additive_residual.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from sourceformer import SourceFormerK3

RESULTS = Path("phase5_results")
SOURCES = ["text", "table", "kg"]
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-1.0, 1.0001, 0.01), 3)


def macro(pred, g, k=3):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def sg_logits(emb):
    outs = []
    for s in SEEDS:
        ck = torch.load(Path("checkpoints") / f"sourceformer_k3_seed{s}_best.pt",
                        map_location="cpu", weights_only=False)
        m = SourceFormerK3(dropout=0.2)
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        with torch.no_grad():
            outs.append(m(torch.from_numpy(emb).float()).numpy())
    return np.mean(outs, 0)


def boot_diff(pa, pb, g, rng, b=B):
    n = len(g)
    idx = rng.integers(0, n, size=(b, n))
    d = np.array([macro(pa[r], g[r]) - macro(pb[r], g[r]) for r in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


# ── data ─────────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
Lt_all = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])
gt_all = np.array([r["true_label"] for r in res])
ok_t = (Lt_all > -1e8).all(1)
Lt, gt = Lt_all[ok_t], gt_all[ok_t]
SGt = log_softmax(sg_logits(np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok_t]))

dev = np.load(RESULTS / "prefrag_conf_dev_scores.npz")
ok_d = (dev["scores"] > -1e8).all(1)
Ld, gd = dev["scores"][ok_d], dev["gold"][ok_d]
SGd = log_softmax(sg_logits(dev["embs"][ok_d].astype(np.float32)))

# standardise likelihood per source using DEV statistics only
mu, sd = Ld.mean(0), Ld.std(0)
Zd, Zt = (Ld - mu) / sd, (Lt - mu) / sd

rng = np.random.default_rng(SEED)
sg_arg_t = SGt.argmax(1)
base_t = macro(sg_arg_t, gt)
print(f"dev  n={len(gd)}  SG {macro(SGd.argmax(1), gd):.4f}")
print(f"test n={len(gt)}  SG {base_t:.4f}  (published 0.7415)")

out = {"n_dev": int(len(gd)), "n_test": int(len(gt)),
       "sourcegate_test_macro": base_t}


def pick_lambda(SG, Z, g):
    """Scalar lambda maximising macro; ties broken toward 0 (the incumbent)."""
    best_m, best_l = -1.0, 0.0
    for lam in GRID:
        m = macro((SG + lam * Z).argmax(1), g)
        if m > best_m + 1e-12 or (abs(m - best_m) <= 1e-12 and abs(lam) < abs(best_l)):
            best_m, best_l = m, lam
    return float(best_l), float(best_m)


def pick_lambda_per_source(SG, Z, g, rounds=3):
    """Coordinate ascent over three lambdas, started at 0 (= SourceGate)."""
    lam = np.zeros(3)
    for _ in range(rounds):
        for j in range(3):
            best_m, best_v = -1.0, lam[j]
            for v in GRID:
                trial = lam.copy()
                trial[j] = v
                m = macro((SG + Z * trial).argmax(1), g)
                if m > best_m + 1e-12 or (abs(m - best_m) <= 1e-12 and abs(v) < abs(best_v)):
                    best_m, best_v = m, v
            lam[j] = best_v
    return lam, macro((SG + Z * lam).argmax(1), g)


# ── P1 dev-fit / test-frozen ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("P1  lambda fitted on DEV, applied once to frozen TEST")
print("=" * 78)
lam1, devm = pick_lambda(SGd, Zd, gd)
pred1 = (SGt + lam1 * Zt).argmax(1)
d1 = boot_diff(pred1, sg_arg_t, gt, rng)
print(f"  scalar      lambda*={lam1:+.3f}  dev {devm:.4f}  test {macro(pred1, gt):.4f}"
      f"   vs SG {d1['mean']:+.4f}  CI [{d1['ci95'][0]:+.4f}, {d1['ci95'][1]:+.4f}]"
      f"  p={d1['p_two_sided']:.4f}")

lamv1, devmv = pick_lambda_per_source(SGd, Zd, gd)
predv1 = (SGt + Zt * lamv1).argmax(1)
dv1 = boot_diff(predv1, sg_arg_t, gt, rng)
print(f"  per-source  lambda*={np.round(lamv1, 3).tolist()}  dev {devmv:.4f}  "
      f"test {macro(predv1, gt):.4f}   vs SG {dv1['mean']:+.4f}  "
      f"CI [{dv1['ci95'][0]:+.4f}, {dv1['ci95'][1]:+.4f}]  p={dv1['p_two_sided']:.4f}")

out["P1_dev_fit"] = {
    "scalar": {"lambda": lam1, "dev_macro": devm, "test_macro": macro(pred1, gt), **d1},
    "per_source": {"lambda": np.round(lamv1, 3).tolist(), "dev_macro": devmv,
                   "test_macro": macro(predv1, gt), **dv1}}

# ── P2 cross-fit within test ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("P2  cross-fit within TEST (lambda from one half, applied to the other)")
print("=" * 78)
half = rng.permutation(len(gt))
A, Bx = half[: len(gt) // 2], half[len(gt) // 2:]
pred2 = np.zeros(len(gt), dtype=int)
lams = {}
for name, fit, app in (("A->B", A, Bx), ("B->A", Bx, A)):
    lam, _ = pick_lambda(SGt[fit], Zt[fit], gt[fit])
    lams[name] = lam
    pred2[app] = (SGt[app] + lam * Zt[app]).argmax(1)
d2 = boot_diff(pred2, sg_arg_t, gt, rng)
print(f"  scalar      lambdas {lams}  test {macro(pred2, gt):.4f}"
      f"   vs SG {d2['mean']:+.4f}  CI [{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}]"
      f"  p={d2['p_two_sided']:.4f}")

pred2v = np.zeros(len(gt), dtype=int)
lamsv = {}
for name, fit, app in (("A->B", A, Bx), ("B->A", Bx, A)):
    lam, _ = pick_lambda_per_source(SGt[fit], Zt[fit], gt[fit])
    lamsv[name] = np.round(lam, 3).tolist()
    pred2v[app] = (SGt[app] + Zt[app] * lam).argmax(1)
d2v = boot_diff(pred2v, sg_arg_t, gt, rng)
print(f"  per-source  lambdas {lamsv}  test {macro(pred2v, gt):.4f}"
      f"   vs SG {d2v['mean']:+.4f}  CI [{d2v['ci95'][0]:+.4f}, {d2v['ci95'][1]:+.4f}]"
      f"  p={d2v['p_two_sided']:.4f}")

out["P2_crossfit"] = {
    "scalar": {"lambdas": lams, "test_macro": macro(pred2, gt), **d2},
    "per_source": {"lambdas": lamsv, "test_macro": macro(pred2v, gt), **d2v}}

# ── sensitivity: test macro across the whole lambda grid ─────────────────────
curve = [(float(l), macro((SGt + l * Zt).argmax(1), gt)) for l in GRID[::5]]
best_l, best_m = max(curve, key=lambda t: t[1])
out["oracle_lambda_on_test"] = {"lambda": best_l, "macro": best_m,
                                "note": "test-fitted upper bound, not achievable"}
out["lambda_curve_test"] = curve
print(f"\n  oracle lambda on test: {best_l:+.2f} -> {best_m:.4f} "
      f"(upper bound; SG alone {base_t:.4f})")

json.dump(out, open(RESULTS / "additive_residual.json", "w"), indent=2)
print("\n  A1 cashable     -> test macro > 0.7415 with CI excluding 0, both protocols")
print("  A2 not cashable -> lambda ~ 0 or gain within noise; claim is informational")
print(f"\nSaved -> {RESULTS / 'additive_residual.json'}")
