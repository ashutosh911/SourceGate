"""
script_54_subsumption_strength.py  --  hardening the subsumption claim

THE CLAIM
  Reader generation-confidence adds nothing to source routing once embedding
  similarity to the retrieved content is available:

      mmRAG K=3   reader beyond similarity  +0.0085  p=0.26
      OTT-QA K=2  reader beyond similarity  +0.0030  p=0.75
      similarity beyond SourceGate          +0.0446  p=0.0002  /  +0.0292  p=0.0004

  A null is only as good as its power, and "p > 0.05" is the weakest possible
  way to state one.  A reviewer's first move will be "your test is
  underpowered".  Four analyses answer that properly.

  T1  EQUIVALENCE (TOST).  Rather than failing to reject zero, bound the
      effect: report the largest reader contribution consistent with the data.
      A one-sided upper confidence limit is the honest headline number and
      turns "we found nothing" into "we exclude anything above X".

  T2  ORACLE CEILING.  Fit the combiner ON the test set, choosing lambda to
      maximise test macro directly.  No honest method can beat this.  If even
      a cheating combiner extracts little from the reader once similarity is
      present, the null is a property of the signal, not of our power or our
      architecture.  This is the single strongest form of the argument.

  T3  REDUNDANCY, DIRECTLY.  How much of the reader's score is linearly
      predictable from similarity?  How often do their argmaxes agree?  If the
      reader is largely a noisy restatement of similarity, subsumption has a
      mechanism rather than only a statistic.

  T4  PER-SOURCE.  If the reader helps anywhere, it should show up in one
      source type.  Reporting the breakdown pre-empts "you averaged away a real
      effect on tables/KG" -- which the parallel Qwen work specifically saw.

  A permutation null (reader features shuffled across queries) calibrates how
  much apparent marginal value the fitting procedure invents from noise, so T2
  is read against that floor rather than against zero.

USAGE
  conda activate chestx && python script_54_subsumption_strength.py

OUTPUT
  phase5_results/subsumption_strength.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from sourceformer import SourceFormerK3

RESULTS = Path("phase5_results")
OTT = Path("phase8_results_ottqa")
SOURCES = ["text", "table", "kg"]
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-3, 5.001, 0.02), 3)
N_PERM = 200


def macro(pred, g, k):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def fit_lambdas(SG, Zs, g, k, rounds=4):
    lam = np.zeros(len(Zs))
    for _ in range(rounds):
        for j in range(len(Zs)):
            bm, bv = -1.0, lam[j]
            for v in GRID:
                t = lam.copy()
                t[j] = v
                m = macro((SG + sum(t[i] * Zs[i] for i in range(len(Zs)))).argmax(1), g, k)
                if m > bm + 1e-12 or (abs(m - bm) <= 1e-12 and abs(v) < abs(bv)):
                    bm, bv = m, v
            lam[j] = bv
    return lam


def z(X):
    return (X - X.mean(0)) / X.std(0)


def analyse(tag, SG, SIM, RD, g, k, rng):
    """All four analyses for one benchmark."""
    out = {"n": int(len(g)), "k": k}
    Zs, Zr = z(SIM), z(RD)
    sg_arg = SG.argmax(1)

    # ── T2 oracle ceiling: lambdas fitted directly on test ───────────────────
    lam_s = fit_lambdas(SG, [Zs], g, k)
    p_sim = (SG + lam_s[0] * Zs).argmax(1)
    lam_sr = fit_lambdas(SG, [Zs, Zr], g, k)
    p_both = (SG + lam_sr[0] * Zs + lam_sr[1] * Zr).argmax(1)
    oracle_gain = macro(p_both, g, k) - macro(p_sim, g, k)

    # permutation floor: same fitting, reader shuffled across queries
    perm_gains = []
    for _ in range(N_PERM):
        Zp = Zr[rng.permutation(len(g))]
        lam_p = fit_lambdas(SG, [Zs, Zp], g, k, rounds=2)
        perm_gains.append(
            macro((SG + lam_p[0] * Zs + lam_p[1] * Zp).argmax(1), g, k) - macro(p_sim, g, k))
    perm_gains = np.array(perm_gains)
    out["T2_oracle"] = {
        "sg_macro": macro(sg_arg, g, k),
        "sg_plus_sim_oracle": macro(p_sim, g, k),
        "sg_plus_sim_reader_oracle": macro(p_both, g, k),
        "oracle_reader_gain": float(oracle_gain),
        "permutation_floor_mean": float(perm_gains.mean()),
        "permutation_floor_p95": float(np.percentile(perm_gains, 95)),
        "p_vs_permutation": float((perm_gains >= oracle_gain).mean()),
    }

    # ── T1 equivalence bound on the HONEST (cross-fit) estimate ──────────────
    half = rng.permutation(len(g))
    A, Bx = half[: len(g) // 2], half[len(g) // 2:]
    preds = {}
    for name, keys in (("sim", [0]), ("both", [0, 1])):
        p = np.zeros(len(g), dtype=int)
        for fit, app in ((A, Bx), (Bx, A)):
            Zf, Za = [], []
            for kk in keys:
                X = [SIM, RD][kk]
                mu, sd = X[fit].mean(0), X[fit].std(0)
                Zf.append((X[fit] - mu) / sd)
                Za.append((X[app] - mu) / sd)
            lam = fit_lambdas(SG[fit], Zf, g[fit], k)
            p[app] = (SG[app] + sum(lam[i] * Za[i] for i in range(len(keys)))).argmax(1)
        preds[name] = p
    idx = rng.integers(0, len(g), size=(B, len(g)))
    d = np.array([macro(preds["both"][r], g[r], k) - macro(preds["sim"][r], g[r], k)
                  for r in idx])
    out["T1_equivalence"] = {
        "point": float(d.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "upper_90_one_sided": float(np.percentile(d, 90)),
        "upper_95_one_sided": float(np.percentile(d, 95)),
    }

    # ── T3 redundancy ────────────────────────────────────────────────────────
    r2 = LinearRegression().fit(SIM, RD).score(SIM, RD)
    per_src_r = [float(np.corrcoef(SIM[:, j], RD[:, j])[0, 1]) for j in range(k)]
    out["T3_redundancy"] = {
        "r2_reader_from_similarity": float(r2),
        "per_source_pearson": per_src_r,
        "argmax_agreement": float((SIM.argmax(1) == RD.argmax(1)).mean()),
        "reader_macro": macro(RD.argmax(1), g, k),
        "similarity_macro": macro(SIM.argmax(1), g, k),
    }

    # ── T4 per-source marginal value (cross-fit predictions) ─────────────────
    out["T4_per_source"] = {
        str(j): {"n": int((g == j).sum()),
                 "sg_plus_sim": float((preds["sim"][g == j] == j).mean()),
                 "sg_plus_sim_reader": float((preds["both"][g == j] == j).mean())}
        for j in range(k)}
    return out


rng = np.random.default_rng(SEED)
report = {}

# ── mmRAG K=3 ────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
L = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])
g3 = np.array([r["true_label"] for r in res])
ok = (L > -1e8).all(1)
L, g3 = L[ok], g3[ok]
emb = np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok]
outs = []
for s in SEEDS:
    ck = torch.load(Path("checkpoints") / f"sourceformer_k3_seed{s}_best.pt",
                    map_location="cpu", weights_only=False)
    m = SourceFormerK3(dropout=0.2)
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    m.eval()
    with torch.no_grad():
        outs.append(m(torch.from_numpy(emb).float()).numpy())
SG3 = log_softmax(np.mean(outs, 0))
SIM3 = np.load(RESULTS / "bge_maxsim_scores.npz")["test"]
report["mmrag_k3"] = analyse("mmrag_k3", SG3, SIM3, L, g3, 3, rng)

# ── OTT-QA K=2 ───────────────────────────────────────────────────────────────
SG2_PATH = OTT / "sourcegate_k2_logits_devselected.npy"
if SG2_PATH.exists():
    L2 = np.load(OTT / "prefrag_conf_k2_scores.npy")
    S2 = np.load(OTT / "ctx_similarity_k2.npy")
    g2 = np.array([int(r["label"]) for r in json.load(open(OTT / "ottqa_k2_test.json"))])
    SG2 = np.load(SG2_PATH)
    ok2 = (L2 > -1e8).all(1)
    report["ottqa_k2"] = analyse("ottqa_k2", SG2[ok2], S2[ok2], L2[ok2], g2[ok2], 2, rng)
else:
    print(f"  skipping OTT-QA: {SG2_PATH} not found (run script_53 first)")

for name, r in report.items():
    print("=" * 78)
    print(f"{name}   n={r['n']}  K={r['k']}")
    print("=" * 78)
    t2 = r["T2_oracle"]
    print(f"  T2 ORACLE CEILING (lambdas fitted ON test -- upper bound)")
    print(f"     SourceGate                      {t2['sg_macro']:.4f}")
    print(f"     + similarity (oracle)           {t2['sg_plus_sim_oracle']:.4f}")
    print(f"     + similarity + reader (oracle)  {t2['sg_plus_sim_reader_oracle']:.4f}")
    print(f"     oracle reader gain              {t2['oracle_reader_gain']:+.4f}")
    print(f"     permutation floor (mean / p95)  {t2['permutation_floor_mean']:+.4f}"
          f" / {t2['permutation_floor_p95']:+.4f}")
    print(f"     p vs permutation null           {t2['p_vs_permutation']:.3f}")
    t1 = r["T1_equivalence"]
    print(f"  T1 EQUIVALENCE (honest cross-fit estimate)")
    print(f"     point {t1['point']:+.4f}   95% CI [{t1['ci95'][0]:+.4f}, {t1['ci95'][1]:+.4f}]")
    print(f"     one-sided upper limit: 90% {t1['upper_90_one_sided']:+.4f}"
          f"   95% {t1['upper_95_one_sided']:+.4f}")
    t3 = r["T3_redundancy"]
    print(f"  T3 REDUNDANCY")
    print(f"     R^2 of reader scores from similarity  {t3['r2_reader_from_similarity']:.4f}")
    print(f"     per-source pearson                    "
          f"{[round(x, 3) for x in t3['per_source_pearson']]}")
    print(f"     argmax agreement                      {t3['argmax_agreement']:.3f}")
    print(f"     reader macro {t3['reader_macro']:.4f}   similarity macro "
          f"{t3['similarity_macro']:.4f}")
    print(f"  T4 PER-SOURCE (cross-fit)")
    for j, v in r["T4_per_source"].items():
        print(f"     source {j}  n={v['n']:<5} +sim {v['sg_plus_sim']:.4f}"
              f"   +sim+reader {v['sg_plus_sim_reader']:.4f}"
              f"   delta {v['sg_plus_sim_reader'] - v['sg_plus_sim']:+.4f}")

json.dump(report, open(RESULTS / "subsumption_strength.json", "w"), indent=2)
print(f"\nSaved -> {RESULTS / 'subsumption_strength.json'}")
