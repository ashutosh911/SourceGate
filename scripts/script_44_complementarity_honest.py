"""
script_44_complementarity_honest.py  --  does reader likelihood carry source
                                         information a trained query router
                                         does not already have?

THE CLAIM UNDER TEST
  Corrected, PrefRAG-Conf routes at 0.588 and SourceGate at 0.742.  The paper's
  story -- that generation confidence collapses under a source-format and
  context-length confound -- is false.  The question that replaces it is
  whether the reader's signal is merely WEAKER than a learned query router or
  actually REDUNDANT with it.

  Those two have opposite implications.  Redundant means generation-confidence
  routing is a strictly dominated design: everything it knows is already in the
  query embedding, and paying K reader forward passes buys nothing.
  Complementary means the two signals see different things, and the right
  architecture combines them.

  A first pass (script_42) found likelihood adds +0.035 macro over SourceGate's
  logits, CI [+0.011, +0.059], p=0.003 -- but fitted by cross-validation ON the
  test set, which cannot be published.

PROTOCOL
  Two independent estimates, reported side by side.  The claim is believed only
  if both agree.

  P1  dev-fit / test-frozen.  Combiner fitted once on the 766 dev queries,
      evaluated once on the 1284 test queries.
      CAVEAT, stated in the output: SourceGate's checkpoint was early-stopped on
      dev macro, so dev is not perfectly clean with respect to SG's logits.
      The contamination is weak (checkpoint selection, not gradient fitting)
      but it is real and it is why P2 exists.

  P2  cross-fit within test.  Split test in half by a seeded partition, fit on
      A predict B and fit on B predict A, concatenate out-of-fold predictions.
      Clean with respect to SourceGate, which never saw test.

WHAT DISTINGUISHES THE MECHANISMS
  If likelihood helps only because it detects retrieval failure, then replacing
  it with a predicted-retrieval-hit feature (trained on the train split, so
  honest) should recover the same gain.  Feature set 3 tests exactly that, and
  set 4 tests whether likelihood still adds anything on top of it.

    1  sg_only                SourceGate logits
    2  sg_plus_likelihood     + reader scores and their contrasts
    3  sg_plus_phit           + predicted per-source retrieval success
    4  sg_plus_both           + both

USAGE
  conda activate chestx && python script_44_complementarity_honest.py

OUTPUT
  phase5_results/complementarity_honest.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sourceformer import SourceFormerK3

RESULTS = Path("phase5_results")
SOURCES = ["text", "table", "kg"]
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000


def macro(pred, g, k=3):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def boot_diff(pa, pb, g, rng, b=B):
    n = len(g)
    idx = rng.integers(0, n, size=(b, n))
    d = np.array([macro(pa[r], g[r]) - macro(pb[r], g[r]) for r in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


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


def lik_feats(L):
    return np.column_stack([L, L[:, 0] - L[:, 1], L[:, 0] - L[:, 2],
                            L[:, 1] - L[:, 2], L - L.mean(1, keepdims=True)])


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


# ── test side ────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
Lt_all = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])
gt_all = np.array([r["true_label"] for r in res])
ok_t = (Lt_all > -1e8).all(1)
Lt, gt = Lt_all[ok_t], gt_all[ok_t]
emb_t = np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok_t]
SGt = sg_logits(emb_t)

# ── dev side ─────────────────────────────────────────────────────────────────
dev = np.load(RESULTS / "prefrag_conf_dev_scores.npz")
Ld_all, gd_all, emb_d_all = dev["scores"], dev["gold"], dev["embs"]
ok_d = (Ld_all > -1e8).all(1)
Ld, gd, emb_d = Ld_all[ok_d], gd_all[ok_d], emb_d_all[ok_d].astype(np.float32)
SGd = sg_logits(emb_d)

print(f"dev  n={len(gd)}  likelihood macro {macro(Ld.argmax(1), gd):.4f}  "
      f"SG macro {macro(SGd.argmax(1), gd):.4f}")
print(f"test n={len(gt)}  likelihood macro {macro(Lt.argmax(1), gt):.4f}  "
      f"SG macro {macro(SGt.argmax(1), gt):.4f}")

# ── predicted retrieval success, fitted on TRAIN only ────────────────────────
tr_emb = np.load("query_emb_cache/train_embs_k3.npy").astype(np.float32)
tr_hit = np.load(RESULTS / "hit_matrix_train_strict.npy").astype(int)
phit_t = np.zeros((len(gt), 3))
phit_d = np.zeros((len(gd), 3))
for j in range(3):
    sc = StandardScaler().fit(tr_emb)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(tr_emb), tr_hit[:, j])
    phit_t[:, j] = clf.predict_proba(sc.transform(emb_t))[:, 1]
    phit_d[:, j] = clf.predict_proba(sc.transform(emb_d))[:, 1]

FEATS = {
    "sg_only":            (lambda S, L, P: S),
    "sg_plus_likelihood": (lambda S, L, P: np.hstack([S, lik_feats(L)])),
    "sg_plus_phit":       (lambda S, L, P: np.hstack([S, P])),
    "sg_plus_both":       (lambda S, L, P: np.hstack([S, lik_feats(L), P])),
}

rng = np.random.default_rng(SEED)
out = {"n_dev": int(len(gd)), "n_test": int(len(gt)),
       "sourcegate_argmax_test": macro(SGt.argmax(1), gt),
       "likelihood_argmax_test": macro(Lt.argmax(1), gt),
       "likelihood_argmax_dev": macro(Ld.argmax(1), gd)}

# ── P1 dev-fit / test-frozen ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("P1  DEV-FIT / TEST-FROZEN   (caveat: SG early-stopped on dev macro)")
print("=" * 78)
p1, preds1 = {}, {}
for name, f in FEATS.items():
    pr = fit_predict(f(SGd, Ld, phit_d), gd, f(SGt, Lt, phit_t))
    preds1[name] = pr
    p1[name] = {"test_macro": macro(pr, gt)}
    print(f"  {name:<24}{macro(pr, gt):.4f}")
for name in FEATS:
    if name == "sg_only":
        continue
    p1[name]["vs_sg_only"] = boot_diff(preds1[name], preds1["sg_only"], gt, rng)
    d = p1[name]["vs_sg_only"]
    print(f"  {name:<24}vs sg_only {d['mean']:+.4f}  "
          f"CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["P1_dev_fit_test_frozen"] = p1

# ── P2 cross-fit within test ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("P2  CROSS-FIT WITHIN TEST   (clean w.r.t. SourceGate)")
print("=" * 78)
half = rng.permutation(len(gt))
A, Bx = half[: len(gt) // 2], half[len(gt) // 2:]
p2, preds2 = {}, {}
for name, f in FEATS.items():
    X = f(SGt, Lt, phit_t)
    pr = np.zeros(len(gt), dtype=int)
    pr[Bx] = fit_predict(X[A], gt[A], X[Bx])
    pr[A] = fit_predict(X[Bx], gt[Bx], X[A])
    preds2[name] = pr
    p2[name] = {"test_macro": macro(pr, gt)}
    print(f"  {name:<24}{macro(pr, gt):.4f}")
for name in FEATS:
    if name == "sg_only":
        continue
    p2[name]["vs_sg_only"] = boot_diff(preds2[name], preds2["sg_only"], gt, rng)
    d = p2[name]["vs_sg_only"]
    print(f"  {name:<24}vs sg_only {d['mean']:+.4f}  "
          f"CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["P2_crossfit_within_test"] = p2

# ── headline comparison against the plain router ─────────────────────────────
print("\n" + "=" * 78)
print("vs the PLAIN SourceGate argmax (the published 0.7415)")
print("=" * 78)
sg_arg = SGt.argmax(1)
head = {}
for tag, preds in (("P1", preds1), ("P2", preds2)):
    d = boot_diff(preds["sg_plus_likelihood"], sg_arg, gt, rng)
    head[tag] = {"macro": macro(preds["sg_plus_likelihood"], gt), **d}
    print(f"  {tag} sg+likelihood {macro(preds['sg_plus_likelihood'], gt):.4f}  "
          f"vs {macro(sg_arg, gt):.4f}   {d['mean']:+.4f}  "
          f"CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["vs_plain_sourcegate"] = head

json.dump(out, open(RESULTS / "complementarity_honest.json", "w"), indent=2)
print("\n  believe the result only if P1 and P2 agree in sign and rough magnitude")
print(f"Saved -> {RESULTS / 'complementarity_honest.json'}")
