"""
script_41_why_likelihood_loses.py  --  decomposing the 15-point gap between
                                       generation-confidence routing (0.588)
                                       and a learned embedding router (0.737)

THE QUESTION
  The original diagnostic -- that reader likelihood collapses under a
  source-format and context-length confound -- is false: corrected, PrefRAG-Conf
  reaches macro 0.588, well above random (0.337), with pick rates 50/20/30
  against base rates 56/28/16.  It does not collapse.  But it still loses by 15
  macro points to a 461K-parameter MLP over the same frozen query embedding it
  never sees.  That gap is the real phenomenon and it has no explanation yet.

  Three candidate mechanisms are testable from artifacts already on disk.

  M1 SUBSUMPTION.  The likelihood carries no information about source identity
     beyond what the frozen embedding already encodes.  If so, adding the three
     likelihood scores to an embedding-only classifier changes nothing, and the
     whole generation-confidence family is not a weaker signal but a redundant
     one.  This is the strongest available claim: it would say the reader's
     opinion is not merely noisy, it is already contained in the query text.

  M2 DECISION RULE.  The information is present in the three scores but argmax
     cannot extract it -- because the scores are on incomparable per-source
     scales, or because their informative structure is in contrasts rather than
     in which is largest.  If a learned readout over the same three numbers
     reaches far above 0.588, the failure is in the decision rule, not in the
     signal, and "confidence routing fails" is the wrong description.

  M3 RETRIEVAL CEILING.  On queries where no source retrieved a relevant chunk,
     every candidate context is irrelevant, so the signal is scoring noise by
     construction.  The scrambled control showed that regime routes at 0.357.
     If the gap concentrates there, the limitation is retrieval, not the reader.

  These are not exclusive and the point is to apportion, not to pick a winner.

PROTOCOL
  Every learned readout is fitted by stratified 5-fold cross-validation over the
  test queries, reporting mean +- sd across folds.  No dev-set likelihood scores
  exist, so CV on test is the honest fallback; it is an UPPER BOUND on what a
  properly held-out readout would achieve, and is labelled as such everywhere.
  The comparison of interest is between feature sets under an identical
  protocol, which the shared folds make fair.

USAGE
  conda activate chestx && python script_41_why_likelihood_loses.py

OUTPUT
  phase5_results/why_likelihood_loses.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RESULTS = Path("phase5_results")
SOURCES = ["text", "table", "kg"]
SEED = 20260926
FOLDS = 5


def macro(pred, g, k=3):
    return float(np.mean([(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]))


def cv_macro(X, y, model="logreg"):
    """Stratified 5-fold CV macro accuracy; returns (mean, sd, oof predictions)."""
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=int)
    per_fold = []
    for tr, te in skf.split(X, y):
        if model == "logreg":
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
            m.fit(sc.transform(X[tr]), y[tr])
            p = m.predict(sc.transform(X[te]))
        else:
            m = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
        oof[te] = p
        per_fold.append(macro(p, y[te]))
    return float(np.mean(per_fold)), float(np.std(per_fold, ddof=1)), oof


# ── data ─────────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
L = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])   # log p(q|c_s)
gold = np.array([r["true_label"] for r in res])
ok = (L > -1e8).all(1)
L, gold = L[ok], gold[ok]

emb = np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok]
hit = np.load(RESULTS / "hit_matrix_test_strict.npy")[ok]          # (n,3) bool
sg = None
sg_path = RESULTS / "sourcegate_decisions_k3.npy"

n = len(gold)
print(f"n = {n}   emb {emb.shape}   hit {hit.shape}")
print(f"gold counts {dict(zip(SOURCES, [int((gold == j).sum()) for j in range(3)]))}")
print(f"raw argmax macro {macro(L.argmax(1), gold):.4f}")

out = {"n": int(n), "protocol": f"stratified {FOLDS}-fold CV on test (UPPER BOUND)",
       "raw_argmax_macro": macro(L.argmax(1), gold)}

# ── M2: is the decision rule the bottleneck? ─────────────────────────────────
print("\n" + "=" * 74)
print("M2  DECISION RULE -- learned readout over the SAME three numbers")
print("=" * 74)
Ld = np.column_stack([L, L[:, 0] - L[:, 1], L[:, 0] - L[:, 2], L[:, 1] - L[:, 2],
                      L - L.mean(1, keepdims=True), L.std(1, keepdims=True)])
m2 = {}
for name, X in [("likelihood_raw3", L), ("likelihood_engineered", Ld)]:
    for mdl in ("logreg", "gbm"):
        mu, sd, _ = cv_macro(X, gold, mdl)
        m2[f"{name}__{mdl}"] = {"macro": mu, "sd": sd}
        print(f"  {name:<24}{mdl:<9}{mu:.4f} +- {sd:.4f}")
out["M2_decision_rule"] = m2

# ── M1: does likelihood add anything to the embedding? ───────────────────────
print("\n" + "=" * 74)
print("M1  SUBSUMPTION -- embedding alone vs embedding + likelihood")
print("=" * 74)
m1 = {}
combos = {"embedding_only": emb,
          "embedding_plus_likelihood": np.hstack([emb, L]),
          "embedding_plus_engineered": np.hstack([emb, Ld])}
oofs = {}
for name, X in combos.items():
    mu, sd, oof = cv_macro(X, gold, "logreg")
    m1[name] = {"macro": mu, "sd": sd}
    oofs[name] = oof
    print(f"  {name:<30}{mu:.4f} +- {sd:.4f}")
delta = m1["embedding_plus_likelihood"]["macro"] - m1["embedding_only"]["macro"]
m1["delta_likelihood_adds"] = delta
print(f"\n  likelihood adds: {delta:+.4f} macro over the embedding alone")

# complementarity: where the embedding router is wrong, is likelihood right?
emb_pred = oofs["embedding_only"]
wrong = emb_pred != gold
lik_pred = L.argmax(1)
rescue = float((lik_pred[wrong] == gold[wrong]).mean())
# chance baseline: among the 2 labels the embedding did not pick
m1["embedding_wrong_n"] = int(wrong.sum())
m1["likelihood_correct_where_embedding_wrong"] = rescue
m1["chance_on_those"] = 0.5
print(f"  embedding router wrong on {wrong.sum()} queries; "
      f"likelihood correct on {rescue:.3f} of them (chance ~0.5 among remaining two)")
out["M1_subsumption"] = m1

# ── M3: retrieval ceiling ────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("M3  RETRIEVAL CEILING -- split by whether ANY source retrieved a hit")
print("=" * 74)
any_hit = hit.any(1)
gold_hit = hit[np.arange(n), gold].astype(bool)
m3 = {}
for name, mask in [("any_source_hit", any_hit), ("no_source_hit", ~any_hit),
                   ("gold_source_hit", gold_hit), ("gold_source_miss", ~gold_hit)]:
    if mask.sum() < 20:
        continue
    m3[name] = {"n": int(mask.sum()),
                "raw_argmax_macro": macro(L.argmax(1)[mask], gold[mask]),
                "embedding_cv_macro": macro(emb_pred[mask], gold[mask])}
    print(f"  {name:<20}n={mask.sum():<6}likelihood {m3[name]['raw_argmax_macro']:.4f}"
          f"   embedding {m3[name]['embedding_cv_macro']:.4f}")
out["M3_retrieval_ceiling"] = m3

json.dump(out, open(RESULTS / "why_likelihood_loses.json", "w"), indent=2)

print("\n" + "=" * 74)
print("READING THE RESULT")
print("  M1 delta ~ 0        -> likelihood is SUBSUMED by the query embedding")
print("  M2 >> 0.588         -> argmax is the bottleneck, not the signal")
print("  M3 gap concentrated -> retrieval failure, not the reader")
print(f"\nSaved -> {RESULTS / 'why_likelihood_loses.json'}")
