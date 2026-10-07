"""
script_60_nll_decomposition_regression.py

R2's strongest surviving criticism: "KG is also the only non-prose source AND
has the lowest oracle recall (0.624 vs 0.748-0.908), and you use r=0.285
(8% of variance) to explain a 40-point drop. Any of these could explain it."

He is right that the original attribution did not separate these. This script
separates them, by regressing the reader's per-(query, source) score on all of
the candidate explanations simultaneously:

    score ~ tokens + is_table + is_kg + similarity + retrieved + dataset

and reporting each term's partial R^2 -- the drop in R^2 when that term alone
is removed from the full model. A term that "could explain the effect" must
survive the presence of the others.

CONVENTIONS
  score      : ceg_scores_d10.npz `real`, stored as NEGATIVE NLL
               (log-likelihood; HIGHER = reader more confident).
  tokens     : Llama-3.1 token count of the scored context. All contexts are
               capped at 800 characters, so this is not raw retrieval length;
               it is the number of tokens the reader actually conditioned on,
               which varies by source because triples and prose tokenise
               differently. This is the length variable that can affect a
               mean-per-token NLL, and is the one the criticism is about.
  similarity : bge_maxsim_scores.npz, max query-chunk cosine per source.
  retrieved  : hit_matrix_test_strict.npy, whether the top-10 from that source
               contains a relevant chunk (the per-query analogue of recall@5).
  dataset    : the query's originating dataset (6 levels), absorbing
               per-dataset difficulty.

Within-query demeaning removes the per-query difficulty constant, which is
what routing cancels anyway (argmax is taken across sources within a query).
Both the raw and demeaned models are reported.

CPU only. Full 1286-query test set.
"""
import json
import re

import numpy as np

SRC = ["text", "table", "kg"]
OUT = "phase5_results/nll_decomposition_regression.json"
rng = np.random.default_rng(20260929)

# ----------------------------------------------------------------- load
z = np.load("phase5_results/ceg_scores_d10.npz", allow_pickle=True)
score, gold = z["real"], z["gold"]          # (Q,3) log-likelihood
Q, S = score.shape

ctx = json.load(open("phase9_results/retrieved_contexts_all.json"))
assert len(ctx) == Q, f"contexts {len(ctx)} vs scores {Q}"

hit = np.load("phase5_results/hit_matrix_test_strict.npy").astype(float)

# The similarity cache holds 1284 rows against 1286 scored queries. It was
# built (script_52:127) over the mask (source_scores > -1e8).all(1), i.e.
# excluding the two queries whose reader scoring hit an error sentinel.
# Reconstruct that exact mask rather than guessing an alignment.
sim_z = np.load("phase5_results/bge_maxsim_scores.npz", allow_pickle=True)
sim_rows = sim_z["test"]
pc = json.load(open("phase5_results/prefrag_conf_results.json"))
Lt_all = np.array([[r["source_scores"][s] for s in SRC] for r in pc])
ok_t = (Lt_all > -1e8).all(1)
assert ok_t.sum() == sim_rows.shape[0], (
    f"mask gives {ok_t.sum()} rows, similarity has {sim_rows.shape[0]}")
sim = np.full((Q, S), np.nan)
sim[ok_t] = sim_rows
print(f"similarity aligned via error-sentinel mask: {ok_t.sum()}/{Q} rows "
      f"({(~ok_t).sum()} sentinel queries excluded)")
# Rows with no similarity are dropped from the regression entirely.
valid = ok_t.copy()

recs = json.load(open("mmrag_test.json"))
ds = np.array([re.match(r"^([a-z]+)", str(r["id"])).group(1) for r in recs])

# ------------------------------------------------------- token lengths
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokens = np.zeros((Q, S))
for i in range(Q):
    c = ctx[str(i)]
    for j, s in enumerate(SRC):
        tokens[i, j] = len(tok(c[s], add_special_tokens=False)["input_ids"])
print("\ncontext tokens at the 800-char budget (what the reader actually saw):")
for j, s in enumerate(SRC):
    print(f"  {s:6s} mean={tokens[:,j].mean():7.1f}  sd={tokens[:,j].std():6.1f}  "
          f"min={tokens[:,j].min():.0f} max={tokens[:,j].max():.0f}")
print("  NOTE: the manuscript cites 7.7 (kg) / 241 (table) / 254 (text) tokens,"
      "\n        which describes untruncated top-10 retrieval, NOT the contexts"
      "\n        actually scored here.")


# --------------------------------------------------------- design matrix
def build(demean):
    rows, y = [], []
    names = None
    for i in range(Q):
        if not valid[i]:
            continue
        sc = score[i].copy()
        tk = tokens[i].copy()
        if demean:
            sc = sc - sc.mean()
        for j in range(S):
            feat = {
                "tokens": tk[j],
                "is_table": 1.0 * (j == 1),
                "is_kg": 1.0 * (j == 2),
                "retrieved": hit[i, j],
                "similarity": sim[i, j],
            }
            for d in sorted(set(ds))[1:]:          # first level is reference
                feat[f"ds_{d}"] = 1.0 * (ds[i] == d)
            if names is None:
                names = list(feat)
            rows.append([feat[k] for k in names])
            y.append(sc[j])
    X = np.asarray(rows, float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)   # z-score
    X = np.column_stack([np.ones(len(X)), X])
    return X, np.asarray(y, float), ["intercept"] + names


def r2(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid.var() / y.var(), beta


res = {"n_queries": int(Q), "n_obs": int(Q * S)}
for demean in (False, True):
    tag = "within_query_demeaned" if demean else "raw"
    X, y, names = build(demean)
    full, beta = r2(X, y)
    print(f"\n{'='*72}\n{tag}:  full model R^2 = {full:.4f}   (n={len(y)})")
    print(f"{'term':<14} {'beta(z)':>10} {'partial R2':>12}   share of model")
    out = {"full_r2": float(full), "terms": {}}
    # group dataset dummies together
    groups = {n: [k] for k, n in enumerate(names) if not n.startswith(("ds_", "intercept"))}
    groups["dataset(6 lvl)"] = [k for k, n in enumerate(names) if n.startswith("ds_")]
    for gname, cols in groups.items():
        keep = [k for k in range(X.shape[1]) if k not in cols]
        red, _ = r2(X[:, keep], y)
        part = full - red
        share = part / full if full > 0 else float("nan")
        b = float(np.mean([beta[k] for k in cols]))
        print(f"{gname:<14} {b:>10.4f} {part:>12.4f}   {100*share:5.1f}%")
        out["terms"][gname] = {"beta_z": b, "partial_r2": float(part),
                               "share_of_model_r2": float(share)}
    res[tag] = out

print(f"\n{'='*72}")
print("READ: a candidate explanation must retain partial R^2 with the others")
print("      present. 'is_kg' absorbing the effect would support R2's point;")
print("      'similarity' absorbing it supports the domination account.")
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"wrote {OUT}")
