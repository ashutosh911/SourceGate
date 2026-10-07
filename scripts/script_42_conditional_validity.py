"""
script_42_conditional_validity.py  --  is reader likelihood a CONDITIONALLY
                                       valid routing signal?

THE HYPOTHESIS THIS TESTS
  script_41 found the 15-point gap is not where the paper says it is.  Split
  the test set by whether the correct source's own retrieval returned a
  relevant chunk:

      gold source retrieved a hit (n=1031)   likelihood 0.629   embedding 0.612
      gold source retrieved nothing (n= 253) likelihood 0.474   embedding 0.533

  Reader likelihood scores the evidence that was actually retrieved.  That is a
  measurement of EVIDENCE SUFFICIENCY.  Routing asks a different question --
  which source SHOULD be searched -- and those come apart precisely when
  retrieval fails: a source can be the right one and still return nothing, and
  the reader has no way to tell that from a source that is simply wrong.

  A query-only router is structurally immune to this, because it never looks at
  retrieved content.  If that is the mechanism, it explains the gap without any
  appeal to source format or context length, and it predicts that the ordering
  between the two routers REVERSES on the retrieval-success subset -- which is
  what we see, and what this script tests properly.

WHAT IS TESTED
  T1  Does the conditional reversal survive a paired bootstrap?
  T2  Does likelihood add information beyond a STRONG embedding router?
      script_41's embedding baseline was CV-fitted on 1k queries and reached
      only 0.602, far below the real SourceGate (0.737) trained on 3072.  The
      honest test compares SourceGate's own logits against SourceGate's logits
      plus likelihood, under one shared CV protocol.
  T3  Is retrieval success PREDICTABLE from the query alone?  If so a gate can
      be built and the mechanism becomes actionable rather than diagnostic.
      The predictor is trained on the TRAIN split (hit_matrix_train_strict) and
      applied to test -- no test labels are used to fit it.
  T4  Oracle gate: route by likelihood where the gold source retrieved a hit,
      by SourceGate elsewhere.  Upper bound on what any gate could buy.
  T5  Honest gate: same, using T3's predicted retrieval success.

  T4 is an upper bound and is labelled as such.  T5 is the deployable number.

USAGE
  conda activate chestx && python script_42_conditional_validity.py

OUTPUT
  phase5_results/conditional_validity.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sourceformer import SourceFormerK3

RESULTS = Path("phase5_results")
SOURCES = ["text", "table", "kg"]
SEEDS = [42, 123, 2026]
SEED = 20260926
FOLDS = 5
B = 10000


def macro(pred, g, k=3):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def cv_macro(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    per_fold = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
        per_fold.append(macro(oof[te], y[te]))
    return float(np.mean(per_fold)), float(np.std(per_fold, ddof=1)), oof


def boot_diff(pa, pb, g, rng, b=B):
    """Paired bootstrap of macro(pa) - macro(pb) over queries."""
    n = len(g)
    idx = rng.integers(0, n, size=(b, n))
    d = np.empty(b)
    for i in range(b):
        r = idx[i]
        d[i] = macro(pa[r], g[r]) - macro(pb[r], g[r])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


# ── data ─────────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
L_all = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])
gold_all = np.array([r["true_label"] for r in res])
ok = (L_all > -1e8).all(1)
L, gold = L_all[ok], gold_all[ok]
emb = np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok]
hit = np.load(RESULTS / "hit_matrix_test_strict.npy").astype(bool)[ok]
n = len(gold)
rng = np.random.default_rng(SEED)

# SourceGate logits, mean over the three supervised seeds
def sg_logits(emb_):
    outs = []
    for s in SEEDS:
        p = Path("checkpoints") / f"sourceformer_k3_seed{s}_best.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = SourceFormerK3(dropout=0.2)
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        with torch.no_grad():
            outs.append(m(torch.from_numpy(emb_).float()).numpy())
    return np.mean(outs, 0)


SG = sg_logits(emb)
sg_pred = SG.argmax(1)
lik_pred = L.argmax(1)
gold_hit = hit[np.arange(n), gold]

print(f"n={n}  SourceGate macro {macro(sg_pred, gold):.4f}  "
      f"likelihood macro {macro(lik_pred, gold):.4f}")
print(f"gold-source retrieval hit: {gold_hit.mean():.1%}")

out = {"n": int(n), "sourcegate_macro": macro(sg_pred, gold),
       "likelihood_macro": macro(lik_pred, gold),
       "gold_hit_rate": float(gold_hit.mean())}

# ── T1 conditional reversal, paired bootstrap within each subset ─────────────
print("\n" + "=" * 76)
print("T1  CONDITIONAL REVERSAL  (paired bootstrap, likelihood - SourceGate)")
print("=" * 76)
t1 = {}
for name, mask in [("gold_source_hit", gold_hit), ("gold_source_miss", ~gold_hit)]:
    d = boot_diff(lik_pred[mask], sg_pred[mask], gold[mask], rng)
    t1[name] = {"n": int(mask.sum()),
                "likelihood": macro(lik_pred[mask], gold[mask]),
                "sourcegate": macro(sg_pred[mask], gold[mask]), **d}
    print(f"  {name:<20}n={mask.sum():<6}lik {t1[name]['likelihood']:.4f}  "
          f"SG {t1[name]['sourcegate']:.4f}  diff {d['mean']:+.4f}  "
          f"CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["T1_conditional_reversal"] = t1

# ── T2 does likelihood add beyond a strong router? ───────────────────────────
print("\n" + "=" * 76)
print("T2  INFORMATION BEYOND SourceGate  (shared 5-fold CV on test)")
print("=" * 76)
Ld = np.column_stack([L, L[:, 0] - L[:, 1], L[:, 0] - L[:, 2], L[:, 1] - L[:, 2],
                      L - L.mean(1, keepdims=True)])
t2 = {}
oofs = {}
for name, X in [("sg_logits_only", SG),
                ("sg_logits_plus_likelihood", np.hstack([SG, Ld])),
                ("likelihood_only", Ld)]:
    mu, sd, oof = cv_macro(X, gold)
    t2[name] = {"macro": mu, "sd": sd}
    oofs[name] = oof
    print(f"  {name:<30}{mu:.4f} +- {sd:.4f}")
t2["delta"] = t2["sg_logits_plus_likelihood"]["macro"] - t2["sg_logits_only"]["macro"]
t2["bootstrap"] = boot_diff(oofs["sg_logits_plus_likelihood"], oofs["sg_logits_only"],
                            gold, rng)
print(f"\n  likelihood adds {t2['delta']:+.4f}  CI "
      f"[{t2['bootstrap']['ci95'][0]:+.4f}, {t2['bootstrap']['ci95'][1]:+.4f}]  "
      f"p={t2['bootstrap']['p_two_sided']:.4f}")
out["T2_information_beyond_sourcegate"] = t2

# ── T3 is retrieval success predictable from the query alone? ────────────────
print("\n" + "=" * 76)
print("T3  RETRIEVAL SUCCESS PREDICTABLE FROM QUERY  (fit on TRAIN, apply to test)")
print("=" * 76)
tr_emb = np.load("query_emb_cache/train_embs_k3.npy").astype(np.float32)
tr_hit = np.load(RESULTS / "hit_matrix_train_strict.npy").astype(bool)
print(f"  train {tr_emb.shape}  hit {tr_hit.shape}")
assert len(tr_emb) == len(tr_hit)
t3 = {}
p_hit = np.zeros((n, 3))
for j, s in enumerate(SOURCES):
    sc = StandardScaler().fit(tr_emb)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(tr_emb), tr_hit[:, j])
    p_hit[:, j] = clf.predict_proba(sc.transform(emb))[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(hit[:, j], p_hit[:, j])
    t3[s] = {"test_auc": float(auc), "train_base_rate": float(tr_hit[:, j].mean()),
             "test_base_rate": float(hit[:, j].mean())}
    print(f"  {s:<8}AUC {auc:.4f}   base rate train {tr_hit[:, j].mean():.3f} "
          f"test {hit[:, j].mean():.3f}")
out["T3_retrieval_predictability"] = t3

# ── T4 oracle gate / T5 honest gate ──────────────────────────────────────────
print("\n" + "=" * 76)
print("T4/T5  GATED HYBRID  (likelihood where retrieval is trusted, else SourceGate)")
print("=" * 76)
gates = {}
oracle_gate = gold_hit
gates["T4_oracle_gate"] = np.where(oracle_gate, lik_pred, sg_pred)
# honest gate: trust likelihood when the source it picks is predicted to hit
conf = p_hit[np.arange(n), lik_pred]
for thr in (0.5, 0.6, 0.7, 0.8):
    gates[f"T5_honest_gate_thr{thr}"] = np.where(conf >= thr, lik_pred, sg_pred)

t45 = {}
for name, pred in gates.items():
    d = boot_diff(pred, sg_pred, gold, rng)
    frac = float((pred == lik_pred).mean())
    t45[name] = {"macro": macro(pred, gold), "frac_routed_by_likelihood": frac, **d}
    print(f"  {name:<26}{macro(pred, gold):.4f}  "
          f"(lik used {frac:.1%})  vs SG {d['mean']:+.4f}  "
          f"CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["T4_T5_gated_hybrid"] = t45

json.dump(out, open(RESULTS / "conditional_validity.json", "w"), indent=2)
print(f"\nSaved -> {RESULTS / 'conditional_validity.json'}")
