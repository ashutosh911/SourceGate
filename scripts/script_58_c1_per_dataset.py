"""
script_58_c1_per_dataset.py

Does the source-format confound concentrate in one constituent dataset?

script_57 found, pooled over all 1286 test queries, that the evidence-free
(format) component is 21.2% of the apparent source gap and that no subset
shows C1-style collapse.  Pooling can hide a per-dataset effect: mmRAG's test
split is six datasets with very different source composition, and a confound
living in one of them would be averaged away.

Pre-registered predictions:
  H1 (C1 survives locally): at least one dataset shows (a) a pick distribution
      concentrated near 1.0 on a single source, or (b) an evidence-free source
      span comparable to or larger than its real source span.
  H0: format nuisance stays a minority share of the source gap everywhere, and
      no dataset collapses.

CONVENTION: ceg_scores_d3.npz stores NEGATIVE NLL, higher = better, argmax.
Uses the frozen 3-donor copy so a concurrent 10-donor run cannot change it.
Full test set, no subsetting -- trap #1 does not apply.
CPU only.
"""
import json
import re
import numpy as np

SRC = ["text", "table", "kg"]
OUT = "phase5_results/c1_per_dataset.json"

z = np.load("phase5_results/ceg_scores_d3.npz", allow_pickle=True)
real, neg, gold = z["real"], z["neg"], z["gold"]
Q, S = real.shape

recs = json.load(open("mmrag_test.json"))
assert len(recs) == Q, f"{len(recs)} records vs {Q} rows"
ds = np.array([re.match(r"^([a-z]+)", str(r["id"])).group(1) for r in recs])

# sanity: ceg gold must agree with the canonical labels used elsewhere
pc = json.load(open("phase5_results/prefrag_conf_results.json"))
tl = np.array([r["true_label"] for r in pc])
print(f"gold vs prefrag_conf true_label disagreements: {int((tl != gold).sum())}/{Q}")

hit = np.load("phase5_results/hit_matrix_test_strict.npy")
pick = real.argmax(1)


def demean(x):
    return x - x.mean(1, keepdims=True)


real_dm = demean(real)
neg_dm = demean(neg.mean(2))

rows = {}
print(f"\n{'dataset':<10} {'n':>5} {'gold t/tb/kg':>16} {'picks t/tb/kg':>18} "
      f"{'max':>6} {'acc':>7} {'macro':>7} {'realSpan':>9} {'negSpan':>8} {'share':>7}")
for name in ["ott", "webqsp", "tat", "cwq", "nq", "triviaqa", "ALL"]:
    m = np.ones(Q, bool) if name == "ALL" else (ds == name)
    n = int(m.sum())
    gd = [int((gold[m] == c).sum()) for c in range(S)]
    pd_ = np.array([(pick[m] == c).mean() for c in range(S)])
    acc = float((pick[m] == gold[m]).mean())
    present = [c for c in range(S) if (m & (gold == c)).sum() > 0]
    macro = float(np.mean([(pick[m & (gold == c)] == c).mean() for c in present]))
    rs = real_dm[m].mean(0)
    ns = neg_dm[m].mean(0)
    real_span = float(rs.max() - rs.min())
    neg_span = float(ns.max() - ns.min())
    share = float(neg_span / real_span) if real_span > 1e-9 else float("nan")
    rows[name] = {
        "n": n, "gold_counts": gd, "pick_dist": pd_.tolist(),
        "max_pick_share": float(pd_.max()), "acc": acc, "macro": macro,
        "classes_present": [SRC[c] for c in present],
        "source_effect_real": {SRC[c]: float(rs[c]) for c in range(S)},
        "source_effect_donors": {SRC[c]: float(ns[c]) for c in range(S)},
        "real_span": real_span, "neg_span": neg_span, "nuisance_share": share,
        "gold_retrieved_rate": float((hit[m, gold[m]] == 1).mean()),
    }
    print(f"{name:<10} {n:>5} {'/'.join(f'{g:>4}' for g in gd):>16} "
          f"{'/'.join(f'{p:.3f}' for p in pd_):>18} {pd_.max():>6.3f} "
          f"{acc:>7.4f} {macro:>7.4f} {real_span:>9.4f} {neg_span:>8.4f} "
          f"{share:>7.3f}")

print("\nper-dataset source effects (nats, within-query demeaned):")
for name in ["ott", "webqsp", "tat", "cwq", "nq", "triviaqa"]:
    r = rows[name]
    print(f"  {name:<9} real  " +
          "  ".join(f"{k}={v:+.4f}" for k, v in r['source_effect_real'].items()))
    print(f"  {'':<9} donor " +
          "  ".join(f"{k}={v:+.4f}" for k, v in r['source_effect_donors'].items()))

# --- verdict checks -------------------------------------------------------
collapse = {k: v["max_pick_share"] for k, v in rows.items()
            if k != "ALL" and v["max_pick_share"] >= 0.90}
nuis_dom = {k: v["nuisance_share"] for k, v in rows.items()
            if k != "ALL" and v["nuisance_share"] >= 0.90}
print("\n" + "=" * 70)
print(f"H1a datasets with pick concentration >= 0.90 : {collapse or 'NONE'}")
print(f"H1b datasets where nuisance >= 90% of gap    : {nuis_dom or 'NONE'}")
print("H0 holds if both are NONE.")
print("=" * 70)

res = {"per_dataset": rows,
       "h1a_collapsed_datasets": collapse,
       "h1b_nuisance_dominated_datasets": nuis_dom,
       "verdict": "H0" if not collapse and not nuis_dom else "H1"}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"wrote {OUT}")
