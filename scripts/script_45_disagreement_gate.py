"""
script_45_disagreement_gate.py  --  can reader/router disagreement identify the
                                    queries where routing should be abandoned?

THE REVIEWER COMMENT THIS ADDRESSES
  Reviewer 2: "Retrieving from all five sources with no routing achieves higher
  end-to-end F1 than the proposed router under both readers.  Routing exists to
  avoid searching every corpus..."  That is true and it is the paper's weakest
  practical point: union 0.4422 F1 vs SourceGate 0.4398.

  The paper's existing answer is a confidence-tiered hybrid that thresholds
  SourceGate's OWN softmax confidence.  Cross-fitted, it reaches 0.4408 -- it
  does not beat union.  That is unsurprising: a router's own confidence is not
  independent evidence about whether the router is right.

THE IDEA
  Reader likelihood is independent evidence.  It is a weaker router (0.588 vs
  0.742) but it is wrong in different places -- script_42 found it carries
  information SourceGate does not have.  So use it not to route, but to decide
  whether to trust the route:

      router and reader AGREE     -> search the routed source only
      router and reader DISAGREE  -> fall back to union retrieval

  This has NO tunable parameter.  There is no threshold to select, nothing to
  fit on dev, and no way to overfit it -- which is why it is worth trying even
  though the confidence-tiered version failed.

PREDICTIONS, FIXED BEFORE RUNNING
  G1  Disagreement marks unreliable routing.
      -> F1 under routing is much lower on the disagreement subset than on the
         agreement subset, and union beats routing THERE specifically.
         The gate then beats both pure strategies.
  G2  Disagreement is uninformative about routing correctness.
      -> routing F1 is similar in both subsets, and the gate lands between the
         two pure strategies, buying nothing.

COST, STATED HONESTLY
  The gate needs PrefRAG-Conf's K reader forward passes to compute the
  agreement test, so it costs what PrefRAG-Conf costs (669.5 ms/query, Table
  10) plus retrieval.  It is a quality/cost trade-off point, not a free win,
  and must be reported on the Pareto plot rather than as a drop-in replacement.

USAGE
  conda activate chestx && python script_45_disagreement_gate.py

OUTPUT
  phase5_results/disagreement_gate.json
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path("phase5_results")
SEED = 20260926
B = 10000
PREFIX = "reader_scaling_llama8b_predictions_"


def load(name):
    """id -> record, from a per-query prediction jsonl."""
    p = RESULTS / f"{PREFIX}{name}.jsonl"
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


pref = load("prefrag_conf")
sg = load("phase3_seed42")
joint = load("phase4_seed42")
union = load("no_routing")
oracle = load("oracle")

ids = sorted(set(pref) & set(sg) & set(union) & set(oracle) & set(joint))
n = len(ids)
print(f"queries joined across all strategies: {n}")

f1_sg = np.array([sg[i]["f1"] for i in ids])
f1_joint = np.array([joint[i]["f1"] for i in ids])
f1_union = np.array([union[i]["f1"] for i in ids])
f1_oracle = np.array([oracle[i]["f1"] for i in ids])
f1_pref = np.array([pref[i]["f1"] for i in ids])
pick_sg = np.array([sg[i]["picked_type"] for i in ids])

# PrefRAG-Conf's picked_type in the prediction jsonl is STALE: that file predates
# the chunk-lookup fix and its picks are the old KG-always policy (1183/1286 KG
# against the corrected 645/257/384).  Using it made agreement fall below chance
# and inverted the accuracy split.  Take the picks from the corrected decision
# array instead, aligned by mmrag_test.json order, which the jsonl preserves.
TYPES = ["text", "table", "kg"]
test_ids = [t["id"] for t in json.load(open("mmrag_test.json"))]
dec = np.load(RESULTS / "prefrag_conf_decisions_k3.npy").astype(int)
assert len(dec) == len(test_ids)
pref_pick_by_id = {qid: TYPES[d] for qid, d in zip(test_ids, dec)}
pick_pref = np.array([pref_pick_by_id[i] for i in ids])
stale = np.array([pref[i]["picked_type"] for i in ids])
print(f"  stale prefrag picks     {dict(zip(*np.unique(stale, return_counts=True)))}")
print(f"  corrected prefrag picks {dict(zip(*np.unique(pick_pref, return_counts=True)))}")

agree = pick_sg == pick_pref
print(f"router/reader agreement rate: {agree.mean():.1%}")

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))


def boot(a, b):
    d = (a[idx] - b[idx]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


print("\n" + "=" * 72)
print("IS DISAGREEMENT INFORMATIVE?  (end-to-end F1 by subset)")
print("=" * 72)
print(f"{'subset':<16}{'n':>7}{'routed':>10}{'union':>10}{'oracle':>10}")
sub = {}
for name, m in (("agree", agree), ("disagree", ~agree)):
    sub[name] = {"n": int(m.sum()), "routed_f1": float(f1_sg[m].mean()),
                 "union_f1": float(f1_union[m].mean()),
                 "oracle_f1": float(f1_oracle[m].mean()),
                 "route_correct": float((pick_sg[m] ==
                                         np.array([sg[i]["oracle_type"] for i in ids])[m]).mean())}
    print(f"{name:<16}{m.sum():>7}{f1_sg[m].mean():>10.4f}"
          f"{f1_union[m].mean():>10.4f}{f1_oracle[m].mean():>10.4f}")
print(f"\n  routing accuracy on agree subset    {sub['agree']['route_correct']:.3f}")
print(f"  routing accuracy on disagree subset {sub['disagree']['route_correct']:.3f}")

# ── the gate ─────────────────────────────────────────────────────────────────
gate = np.where(agree, f1_sg, f1_union)
gate_joint = np.where(agree, f1_joint, f1_union)

print("\n" + "=" * 72)
print("PARAMETER-FREE DISAGREEMENT GATE")
print("=" * 72)
strategies = {
    "SourceGate (supervised)": f1_sg,
    "SourceGate (joint)": f1_joint,
    "PrefRAG-Conf [STALE picks, do not cite]": f1_pref,
    "No routing (union)": f1_union,
    "GATE: agree->route, else union": gate,
    "GATE (joint router)": gate_joint,
    "Oracle routing": f1_oracle,
}
for k, v in strategies.items():
    print(f"  {k:<34}{v.mean():.4f}")

out = {"n": n, "agreement_rate": float(agree.mean()), "subsets": sub,
       "mean_f1": {k: float(v.mean()) for k, v in strategies.items()},
       "frac_union_calls": float((~agree).mean())}

print("\n  gate vs each pure strategy:")
comp = {}
for k, v in (("vs SourceGate", f1_sg), ("vs union", f1_union),
             ("vs existing hybrid (0.4408 crossfit)", None)):
    if v is None:
        continue
    d = boot(gate, v)
    comp[k] = d
    print(f"    {k:<16}{d['mean']:+.4f}  CI [{d['ci95'][0]:+.4f}, "
          f"{d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["gate_vs"] = comp
out["reference_existing_hybrid_crossfit_f1"] = 0.4408
out["cost_note"] = ("gate requires PrefRAG-Conf's K reader passes (669.5 ms/query, "
                    "Table 10) to compute agreement; report on the Pareto plot")

json.dump(out, open(RESULTS / "disagreement_gate.json", "w"), indent=2)
print("\n  G1 -> gate beats both pure strategies; disagreement subset is where routing fails")
print("  G2 -> gate lands between them; disagreement is uninformative")
print(f"\nSaved -> {RESULTS / 'disagreement_gate.json'}")
