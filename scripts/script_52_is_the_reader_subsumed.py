"""
script_52_is_the_reader_subsumed.py  --  does generation confidence add
                                         anything over embedding similarity to
                                         the retrieved content?

WHY THIS IS NOW THE CENTRAL QUESTION
  Adding a standardised residual to SourceGate's log-probabilities,
  score_s = log_softmax(SG)_s + lambda_s * z_s, and choosing lambda on held-out
  data, gives on mmRAG test:

      residual = reader likelihood     0.7642 / 0.7592   (p 0.066 / 0.103)
      residual = BGE max-similarity    0.7863 / 0.7763   (p 0.000 / 0.002)
      residual = logreg logits         0.7413 / 0.7370   (null)
      residual = MLP-HardCE logits     0.7435 / 0.7339   (null)
      residual = gaussian noise        0.7198 / 0.7421   (null)

  Two things follow.  The lambda machinery does not manufacture gains, since
  noise and query-only routers yield nothing.  And the useful second signal is
  not the reader: a cosine similarity against the retrieved corpus beats it,
  at zero reader forward passes.

  What remains undetermined is whether the reader contributes ANYTHING once
  similarity is already in the model.  That single comparison decides which
  claim the paper can make:

  S1  SUBSUMED.  SG+sim+reader is no better than SG+sim.
      -> Generation-confidence routing is strictly dominated.  Everything the
         reader knows about source identity is already available from the
         embedding's similarity to the retrieved chunks, so PrefRAG-style
         methods pay K reader forward passes (669.5 ms vs 178.7 ms) for
         information they could have had for free.  This is a clean, strong,
         falsifiable indictment of the whole family -- and unlike the original
         format-confound story, it survives every correction made to date.

  S2  NOT SUBSUMED.  SG+sim+reader beats SG+sim with a CI excluding zero.
      -> The reader carries residual information beyond similarity and the
         complementarity claim stands in weakened, honest form.

PROTOCOLS
  P1  lambdas chosen on dev, applied once to frozen test.
  P2  cross-fit within test (fit on half A, apply to half B, and conversely).
  Per-seed SourceGate results are reported alongside the 3-seed logit ensemble,
  because the paper's headline (0.737 +- 0.004) is a per-seed mean.
  Lambdas for the two-residual model are fitted by coordinate ascent started at
  zero, so the family strictly contains SG+sim.

USAGE
  conda activate chestx && python script_52_is_the_reader_subsumed.py

OUTPUT
  phase5_results/is_the_reader_subsumed.json
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
GRID = np.round(np.arange(-3, 5.001, 0.02), 3)


def macro(pred, g, k=3):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def boot_diff(pa, pb, g, rng, b=B):
    n = len(g)
    idx = rng.integers(0, n, size=(b, n))
    d = np.array([macro(pa[r], g[r]) - macro(pb[r], g[r]) for r in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def sg_logits(emb, seed=None):
    outs = []
    for s in (SEEDS if seed is None else [seed]):
        ck = torch.load(Path("checkpoints") / f"sourceformer_k3_seed{s}_best.pt",
                        map_location="cpu", weights_only=False)
        m = SourceFormerK3(dropout=0.2)
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        with torch.no_grad():
            outs.append(m(torch.from_numpy(emb).float()).numpy())
    return np.mean(outs, 0)


def fit_lambdas(SG, Zs, g, rounds=4):
    """Coordinate ascent over one lambda per residual signal, started at 0."""
    lam = np.zeros(len(Zs))
    for _ in range(rounds):
        for j in range(len(Zs)):
            best_m, best_v = -1.0, lam[j]
            for v in GRID:
                t = lam.copy()
                t[j] = v
                s = SG + sum(t[k] * Zs[k] for k in range(len(Zs)))
                m = macro(s.argmax(1), g)
                if m > best_m + 1e-12 or (abs(m - best_m) <= 1e-12 and abs(v) < abs(best_v)):
                    best_m, best_v = m, v
            lam[j] = best_v
    return lam


def apply_lam(SG, Zs, lam):
    return (SG + sum(lam[k] * Zs[k] for k in range(len(Zs)))).argmax(1)


# ── data ─────────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
Lt_all = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])
gt_all = np.array([r["true_label"] for r in res])
ok_t = (Lt_all > -1e8).all(1)
Lt, gt = Lt_all[ok_t], gt_all[ok_t]
emb_t = np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok_t]

dev = np.load(RESULTS / "prefrag_conf_dev_scores.npz")
ok_d = (dev["scores"] > -1e8).all(1)
Ld, gd = dev["scores"][ok_d], dev["gold"][ok_d]
emb_d = dev["embs"][ok_d].astype(np.float32)

sim = np.load(RESULTS / "bge_maxsim_scores.npz")
Sd, St = sim["dev"], sim["test"]
assert Sd.shape[0] == len(gd) and St.shape[0] == len(gt)

rng = np.random.default_rng(SEED)
out = {"n_dev": int(len(gd)), "n_test": int(len(gt))}


def zpair(Xd, Xt):
    mu, sd = Xd.mean(0), Xd.std(0)
    return (Xd - mu) / sd, (Xt - mu) / sd


MODELS = {
    "SG": [],
    "SG+sim": ["sim"],
    "SG+reader": ["reader"],
    "SG+sim+reader": ["sim", "reader"],
}

# ── P1 dev-fit / test-frozen, on the 3-seed ensemble ─────────────────────────
SGd, SGt = log_softmax(sg_logits(emb_d)), log_softmax(sg_logits(emb_t))
Zsim_d, Zsim_t = zpair(Sd, St)
Zrd_d, Zrd_t = zpair(Ld, Lt)
BANK_D = {"sim": Zsim_d, "reader": Zrd_d}
BANK_T = {"sim": Zsim_t, "reader": Zrd_t}

print(f"dev n={len(gd)}  test n={len(gt)}")
print(f"  SourceGate test macro   {macro(SGt.argmax(1), gt):.4f}")
print(f"  BGE-sim argmax          {macro(St.argmax(1), gt):.4f}")
print(f"  reader argmax           {macro(Lt.argmax(1), gt):.4f}")

print("\n" + "=" * 78)
print("P1  lambdas on DEV -> frozen TEST  (3-seed logit ensemble)")
print("=" * 78)
p1 = {}
for name, keys in MODELS.items():
    if not keys:
        pred = SGt.argmax(1)
        lam = []
    else:
        lam = fit_lambdas(SGd, [BANK_D[k] for k in keys], gd)
        pred = apply_lam(SGt, [BANK_T[k] for k in keys], lam)
    p1[name] = {"lambda": np.round(lam, 3).tolist(), "test_macro": macro(pred, gt),
                "_pred": pred}
    print(f"  {name:<16}lambda={str(np.round(lam, 2).tolist()):<22} test {macro(pred, gt):.4f}")

key = boot_diff(p1["SG+sim+reader"]["_pred"], p1["SG+sim"]["_pred"], gt, rng)
print(f"\n  KEY TEST  (SG+sim+reader) - (SG+sim) = {key['mean']:+.4f}  "
      f"CI [{key['ci95'][0]:+.4f}, {key['ci95'][1]:+.4f}]  p={key['p_two_sided']:.4f}")
d_sim = boot_diff(p1["SG+sim"]["_pred"], p1["SG"]["_pred"], gt, rng)
print(f"  (SG+sim) - SG                        = {d_sim['mean']:+.4f}  "
      f"CI [{d_sim['ci95'][0]:+.4f}, {d_sim['ci95'][1]:+.4f}]  p={d_sim['p_two_sided']:.4f}")
out["P1"] = {k: {kk: vv for kk, vv in v.items() if kk != "_pred"} for k, v in p1.items()}
out["P1"]["key_reader_beyond_sim"] = key
out["P1"]["sim_beyond_sg"] = d_sim

# ── P2 cross-fit within test ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("P2  cross-fit within TEST")
print("=" * 78)
half = rng.permutation(len(gt))
A, Bx = half[: len(gt) // 2], half[len(gt) // 2:]
p2 = {}
for name, keys in MODELS.items():
    if not keys:
        p2[name] = {"test_macro": macro(SGt.argmax(1), gt), "_pred": SGt.argmax(1)}
        print(f"  {name:<16}test {p2[name]['test_macro']:.4f}")
        continue
    pred = np.zeros(len(gt), dtype=int)
    for fit, app in ((A, Bx), (Bx, A)):
        Zf, Za = [], []
        for k in keys:
            X = {"sim": St, "reader": Lt}[k]
            mu, sd = X[fit].mean(0), X[fit].std(0)
            Zf.append((X[fit] - mu) / sd)
            Za.append((X[app] - mu) / sd)
        lam = fit_lambdas(SGt[fit], Zf, gt[fit])
        pred[app] = apply_lam(SGt[app], Za, lam)
    p2[name] = {"test_macro": macro(pred, gt), "_pred": pred}
    print(f"  {name:<16}test {macro(pred, gt):.4f}")

key2 = boot_diff(p2["SG+sim+reader"]["_pred"], p2["SG+sim"]["_pred"], gt, rng)
d_sim2 = boot_diff(p2["SG+sim"]["_pred"], p2["SG"]["_pred"], gt, rng)
print(f"\n  KEY TEST  (SG+sim+reader) - (SG+sim) = {key2['mean']:+.4f}  "
      f"CI [{key2['ci95'][0]:+.4f}, {key2['ci95'][1]:+.4f}]  p={key2['p_two_sided']:.4f}")
print(f"  (SG+sim) - SG                        = {d_sim2['mean']:+.4f}  "
      f"CI [{d_sim2['ci95'][0]:+.4f}, {d_sim2['ci95'][1]:+.4f}]  p={d_sim2['p_two_sided']:.4f}")
out["P2"] = {k: {"test_macro": v["test_macro"]} for k, v in p2.items()}
out["P2"]["key_reader_beyond_sim"] = key2
out["P2"]["sim_beyond_sg"] = d_sim2

# ── per-seed, against the paper's reported baseline ──────────────────────────
print("\n" + "=" * 78)
print("PER-SEED  (paper reports supervised macro 0.737 +- 0.004)")
print("=" * 78)
rows = {k: [] for k in MODELS}
for s in SEEDS:
    SGd_s, SGt_s = log_softmax(sg_logits(emb_d, s)), log_softmax(sg_logits(emb_t, s))
    line = []
    for name, keys in MODELS.items():
        if not keys:
            m = macro(SGt_s.argmax(1), gt)
        else:
            lam = fit_lambdas(SGd_s, [BANK_D[k] for k in keys], gd)
            m = macro(apply_lam(SGt_s, [BANK_T[k] for k in keys], lam), gt)
        rows[name].append(m)
        line.append(f"{name} {m:.4f}")
    print(f"  seed {s}: " + "  ".join(line))
print()
for name in MODELS:
    a = np.array(rows[name])
    print(f"  {name:<16}{a.mean():.4f} +- {a.std(ddof=1):.4f}")
out["per_seed"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                       "per_seed": v} for k, v in rows.items()}

json.dump(out, open(RESULTS / "is_the_reader_subsumed.json", "w"), indent=2)
print("\n  S1 subsumed     -> reader adds nothing over similarity; PrefRAG family dominated")
print("  S2 not subsumed -> reader carries residual information beyond similarity")
print(f"\nSaved -> {RESULTS / 'is_the_reader_subsumed.json'}")
