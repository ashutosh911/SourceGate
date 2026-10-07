"""
script_55_k5_format_selectivity.py  --  PRE-REGISTERED test of format
                                        selectivity at K=5

WRITTEN AND COMMITTED BEFORE THE K=5 READER SCORES EXISTED.
  script_50 was still in its retrieval phase when this file was written; the
  grouping, the directional predictions and the test statistic below were all
  fixed at that point precisely so they could not be chosen after seeing the
  numbers.

THE FINDING BEING TESTED
  Against a cost-free similarity signal computed from the same retrieval, the
  reader's marginal contribution to routing is format-selective:

      mmRAG K=3   table   +0.0224  [+0.0056, +0.0420]  p=0.007
                  text    -0.0223  [-0.0418, -0.0028]  p=0.027
      OTT-QA K=2  table   +0.0366  [+0.0039, +0.0694]  p=0.034
                  passage -0.0183  [-0.0354, -0.0018]  p=0.029

  The same asymmetry appears independently in parallel Qwen experiments
  (table AUC +0.047, text AUC -0.005), i.e. under a different reader and a
  different scoring implementation.

WHY K=5 IS THE DECISIVE TEST
  At K=3 the source types ARE the format families, so "format-selective" and
  "type-selective" cannot be told apart: any per-type effect is trivially also
  a per-format effect.  At K=5 they come apart, because the five types are the
  individual datasets and two pairs share a format:

      PROSE   nq, triviaqa
      TABLE   ott, tat
      KG      kg

  A format effect must therefore agree WITHIN each pair and differ BETWEEN
  pairs.  A per-dataset idiosyncrasy has no reason to respect that structure.
  K=5 also equalises the retrieval budget: each type is one FAISS index, so
  every candidate gets exactly TOP_K chunks with no merging.

PREDICTIONS, FIXED NOW
  F1  Format selectivity.  delta(ott) > 0 and delta(tat) > 0;
      delta(nq) < 0 and delta(triviaqa) < 0.  The grouped contrast
      mean(table) - mean(prose) is positive with a bootstrap CI excluding 0.
  F2  No structure.  Signs disagree within a pair, or the grouped contrast's
      CI contains 0 -> the K=3 and OTT-QA per-type effects were type-specific
      idiosyncrasies, not a format property, and the claim must be narrowed to
      "per source type" with no format interpretation.

  F2 is a real possibility and the honest outcome if it occurs.  KG is excluded
  from the grouped contrast: it is a single unpaired type and its K=3 estimate
  was unstable across splits (+0.024 vs -0.033, n=210), so it is reported but
  not used as evidence either way.

USAGE
  conda activate chestx && python script_55_k5_format_selectivity.py
  (requires phase5_results/prefrag_conf_k5_scores.npz from script_50)

OUTPUT
  phase5_results/k5_format_selectivity.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from sourceformer import SourceFormerK5, SOURCE_TYPES_K5

RESULTS = Path("phase5_results")
SCORES = RESULTS / "prefrag_conf_k5_scores.npz"
SEEDS = [7, 99, 314]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-3, 5.001, 0.02), 3)
K = 5

PROSE = ["nq", "triviaqa"]
TABLE = ["ott", "tat"]
UNPAIRED = ["kg"]


def macro(pred, g, k=K):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def soft_target_k5(item):
    v = np.array([float(item["dataset_score"].get(t, 0.0)) for t in SOURCE_TYPES_K5],
                 dtype=np.float32)
    return None if v.sum() == 0 else v / v.sum()


def fit_lambdas(SG, Zs, g, rounds=4):
    lam = np.zeros(len(Zs))
    for _ in range(rounds):
        for j in range(len(Zs)):
            bm, bv = -1.0, lam[j]
            for v in GRID:
                t = lam.copy()
                t[j] = v
                m = macro((SG + sum(t[i] * Zs[i] for i in range(len(Zs)))).argmax(1), g)
                if m > bm + 1e-12 or (abs(m - bm) <= 1e-12 and abs(v) < abs(bv)):
                    bm, bv = m, v
            lam[j] = bv
    return lam


def crossfit(SG, Xs, g, rng):
    n = len(g)
    h = rng.permutation(n)
    A, Bx = h[: n // 2], h[n // 2:]
    p = np.zeros(n, dtype=int)
    for fit, app in ((A, Bx), (Bx, A)):
        Zf, Za = [], []
        for X in Xs:
            mu, sd = X[fit].mean(0), X[fit].std(0)
            Zf.append((X[fit] - mu) / sd)
            Za.append((X[app] - mu) / sd)
        lam = fit_lambdas(SG[fit], Zf, g[fit])
        p[app] = (SG[app] + sum(lam[i] * Za[i] for i in range(len(Za)))).argmax(1)
    return p


# ── data ─────────────────────────────────────────────────────────────────────
z = np.load(SCORES)
L, SIM = z["test"], z["sim_test"]
data = json.load(open("mmrag_test.json"))
keep, lab = [], []
for i, it in enumerate(data):
    s = soft_target_k5(it)
    if s is not None:
        keep.append(i)
        lab.append(int(np.argmax(s)))
gold = np.array(lab)
emb = np.load("query_emb_cache/k5_test_embs_prefrag.npy").astype(np.float32)
assert len(emb) == len(gold) == len(L)

outs = []
for s in SEEDS:
    ck = torch.load(Path("checkpoints") / f"sourceformer_k5_seed{s}_best.pt",
                    map_location="cpu", weights_only=False)
    m = SourceFormerK5()
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    m.eval()
    with torch.no_grad():
        outs.append(m(torch.from_numpy(emb).float()).numpy())
SG = log_softmax(np.mean(outs, 0))

ok = (L > -1e8).all(1)
L, SIM, SG, g = L[ok], SIM[ok], SG[ok], gold[ok]
rng = np.random.default_rng(SEED)

print(f"K=5 test n={len(g)}   chance {1/K:.3f}")
print(f"  SourceGate-K5      {macro(SG.argmax(1), g):.4f}")
print(f"  similarity argmax  {macro(SIM.argmax(1), g):.4f}")
print(f"  reader argmax      {macro(L.argmax(1), g):.4f}")

p_sim = crossfit(SG, [SIM], g, rng)
p_both = crossfit(SG, [SIM, L], g, rng)
print(f"\n  SG+sim         {macro(p_sim, g):.4f}")
print(f"  SG+sim+reader  {macro(p_both, g):.4f}")

out = {"n": int(len(g)),
       "sourcegate_k5": macro(SG.argmax(1), g),
       "similarity_argmax": macro(SIM.argmax(1), g),
       "reader_argmax": macro(L.argmax(1), g),
       "sg_plus_sim": macro(p_sim, g),
       "sg_plus_sim_reader": macro(p_both, g),
       "per_source": {}}

print("\n" + "=" * 80)
print("PER-SOURCE MARGINAL VALUE OF THE READER (cross-fit)")
print("=" * 80)
print(f"  {'source':<11}{'group':<8}{'n':>6}{'+sim':>9}{'+sim+rdr':>10}{'delta':>9}"
      f"{'95% CI':>22}{'p':>8}")
deltas = {}
for c, name in enumerate(SOURCE_TYPES_K5):
    m = g == c
    n = int(m.sum())
    if n == 0:
        continue
    idx = rng.integers(0, n, size=(B, n))
    a, b = p_sim[m], p_both[m]
    d = (b[idx] == c).mean(1) - (a[idx] == c).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    grp = "PROSE" if name in PROSE else ("TABLE" if name in TABLE else "-")
    deltas[name] = float((b == c).mean() - (a == c).mean())
    out["per_source"][name] = {"group": grp, "n": n,
                               "sg_plus_sim": float((a == c).mean()),
                               "sg_plus_sim_reader": float((b == c).mean()),
                               "delta": deltas[name],
                               "ci95": [float(lo), float(hi)], "p": pv}
    print(f"  {name:<11}{grp:<8}{n:>6}{(a == c).mean():>9.4f}{(b == c).mean():>10.4f}"
          f"{deltas[name]:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{pv:>8.3f}")

# ── the pre-registered grouped contrast ──────────────────────────────────────
print("\n" + "=" * 80)
print("PRE-REGISTERED CONTRAST:  mean(TABLE) - mean(PROSE)")
print("=" * 80)
n_all = len(g)
idx = rng.integers(0, n_all, size=(B, n_all))
cons = np.empty(B)
for bi in range(B):
    r = idx[bi]
    gr, ar, br = g[r], p_sim[r], p_both[r]
    def dl(names):
        vals = []
        for nm in names:
            c = SOURCE_TYPES_K5.index(nm)
            msk = gr == c
            if msk.sum():
                vals.append((br[msk] == c).mean() - (ar[msk] == c).mean())
        return np.mean(vals) if vals else np.nan
    cons[bi] = dl(TABLE) - dl(PROSE)
cons = cons[~np.isnan(cons)]
lo, hi = np.percentile(cons, [2.5, 97.5])
pv = float(2 * min((cons <= 0).mean(), (cons >= 0).mean()))
point = np.mean([deltas[n] for n in TABLE]) - np.mean([deltas[n] for n in PROSE])
print(f"  TABLE mean delta  {np.mean([deltas[n] for n in TABLE]):+.4f} "
      f"({', '.join(f'{n} {deltas[n]:+.4f}' for n in TABLE)})")
print(f"  PROSE mean delta  {np.mean([deltas[n] for n in PROSE]):+.4f} "
      f"({', '.join(f'{n} {deltas[n]:+.4f}' for n in PROSE)})")
print(f"  contrast          {point:+.4f}   CI [{lo:+.4f}, {hi:+.4f}]   p={pv:.4f}")
if UNPAIRED[0] in deltas:
    print(f"  (kg, unpaired, reported not used: {deltas[UNPAIRED[0]]:+.4f})")

signs_ok = all(deltas[n] > 0 for n in TABLE) and all(deltas[n] < 0 for n in PROSE)
out["contrast"] = {"point": float(point), "ci95": [float(lo), float(hi)], "p": pv,
                   "within_pair_signs_consistent": bool(signs_ok),
                   "table_mean": float(np.mean([deltas[n] for n in TABLE])),
                   "prose_mean": float(np.mean([deltas[n] for n in PROSE]))}
print(f"\n  within-pair sign consistency: {'YES' if signs_ok else 'NO'}")
print(f"  F1 supported: {'YES' if signs_ok and lo > 0 else 'NO'}")

json.dump(out, open(RESULTS / "k5_format_selectivity.json", "w"), indent=2)
print(f"\nSaved -> {RESULTS / 'k5_format_selectivity.json'}")
