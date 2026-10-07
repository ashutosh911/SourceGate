"""
script_46_cheap_second_opinion.py  --  does the disagreement gate need the
                                       reader, or will a cheap second opinion do?

WHERE THIS COMES FROM
  script_45: when SourceGate and the reader's likelihood pick the same source,
  routing accuracy is 0.823 and routed F1 (0.3883) beats union (0.3841).  When
  they disagree, routing accuracy falls to 0.571 and union wins by 6 F1 points.
  Gating on that -- route when they agree, union when they do not -- gives
  0.3668 F1: +0.0260 over SourceGate (p<0.0001), level with union (p=0.59),
  and above oracle single-source routing (0.3636).  No tunable parameter.

  That answers Reviewer 2's strongest practical objection -- union beats the
  router, so why route -- with "you can have union-level answer quality while
  searching a single source on 55% of queries".

  Except the agreement test itself costs K reader forward passes (669.5 ms,
  Table 10) against union's 178.7 ms.  As it stands the gate buys quality parity
  at a 3.7x latency penalty, which is not a practical argument.

THE QUESTION
  Is the reader doing the work, or would ANY independent second opinion do?
  Every alternative below is essentially free at inference:

    bge_confidence   already computed during retrieval (embedding similarity)
    logistic_reg     a linear probe on the same frozen embedding
    mlp_hardce       a hard-label MLP on the same frozen embedding
    joint            the jointly-trained SourceGate checkpoint

  Two outcomes, both publishable and each pointing somewhere different:

  C1  Cheap proxies gate as well as the reader.
      -> the mechanism is ENSEMBLE DISAGREEMENT, not reader knowledge.  The
         practical claim is strong (union-level F1, one source searched most of
         the time, no reader passes) and the diagnostic claim about generation
         confidence gets no support.

  C2  The reader gates better than every cheap proxy.
      -> the reader sees something the embedding family cannot, which supports
         the complementarity result in script_42/44 and justifies its cost on a
         quality/cost Pareto plot.

  Reported: agreement rate, routing accuracy split, gate F1, bootstrap CI vs
  union and vs SourceGate, and the fraction of queries needing a union search.

USAGE
  conda activate chestx && python script_46_cheap_second_opinion.py

OUTPUT
  phase5_results/cheap_second_opinion.json
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path("phase5_results")
TYPES = ["text", "table", "kg"]
SEED = 20260926
B = 10000
PREFIX = "reader_scaling_llama8b_predictions_"


def load_f1(name):
    out = {}
    with open(RESULTS / f"{PREFIX}{name}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


test = json.load(open("mmrag_test.json"))
ids = [t["id"] for t in test]
n = len(ids)

sg = load_f1("phase3_seed42")
union = load_f1("no_routing")
oracle = load_f1("oracle")
assert all(i in sg and i in union for i in ids)

f1_sg = np.array([sg[i]["f1"] for i in ids])
f1_union = np.array([union[i]["f1"] for i in ids])
f1_oracle = np.array([oracle[i]["f1"] for i in ids])
pick_sg = np.array([TYPES.index(sg[i]["picked_type"]) for i in ids])
gold_type = np.array([TYPES.index(sg[i]["oracle_type"]) for i in ids])

# candidate second opinions, all aligned to mmrag_test.json order
second = {
    "reader_likelihood (669.5 ms)": np.load(RESULTS / "prefrag_conf_decisions_k3.npy"),
    "bge_confidence (free)":        np.load(RESULTS / "confidence_decisions_k3.npy"),
    "logistic_regression (free)":   np.load(RESULTS / "lr_decisions_k3.npy"),
    "mlp_hardce (free)":            np.load(RESULTS / "mlp_hardce_decisions_k3.npy"),
}
second = {k: v.astype(int) for k, v in second.items() if len(v) == n}

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))


def boot(a, b):
    d = (a[idx] - b[idx]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


print(f"n={n}   SourceGate F1 {f1_sg.mean():.4f}   union F1 {f1_union.mean():.4f}"
      f"   oracle-routing F1 {f1_oracle.mean():.4f}")
print("\n" + "=" * 94)
print(f"{'second opinion':<30}{'agree%':>8}{'acc|ag':>8}{'acc|dis':>9}"
      f"{'gateF1':>9}{'vs union':>11}{'p':>8}{'union%':>8}")
print("=" * 94)

out = {"n": n, "f1_sourcegate": float(f1_sg.mean()),
       "f1_union": float(f1_union.mean()), "f1_oracle": float(f1_oracle.mean()),
       "gates": {}}

for name, pick2 in second.items():
    agree = pick_sg == pick2
    gate = np.where(agree, f1_sg, f1_union)
    d_union = boot(gate, f1_union)
    d_sg = boot(gate, f1_sg)
    rec = {
        "agreement_rate": float(agree.mean()),
        "routing_acc_agree": float((pick_sg[agree] == gold_type[agree]).mean()),
        "routing_acc_disagree": float((pick_sg[~agree] == gold_type[~agree]).mean()),
        "gate_f1": float(gate.mean()),
        "frac_union_searches": float((~agree).mean()),
        "vs_union": d_union, "vs_sourcegate": d_sg,
    }
    out["gates"][name] = rec
    print(f"{name:<30}{agree.mean():>7.1%}{rec['routing_acc_agree']:>8.3f}"
          f"{rec['routing_acc_disagree']:>9.3f}{gate.mean():>9.4f}"
          f"{d_union['mean']:>+11.4f}{d_union['p_two_sided']:>8.3f}"
          f"{(~agree).mean():>8.1%}")

# unanimity of the three free signals
free = [v for k, v in second.items() if "free" in k]
if len(free) >= 2:
    unan = np.ones(n, dtype=bool)
    for v in free:
        unan &= (pick_sg == v)
    gate = np.where(unan, f1_sg, f1_union)
    d_union, d_sg = boot(gate, f1_union), boot(gate, f1_sg)
    out["gates"]["unanimous_free_signals"] = {
        "agreement_rate": float(unan.mean()),
        "routing_acc_agree": float((pick_sg[unan] == gold_type[unan]).mean()),
        "routing_acc_disagree": float((pick_sg[~unan] == gold_type[~unan]).mean()),
        "gate_f1": float(gate.mean()), "frac_union_searches": float((~unan).mean()),
        "vs_union": d_union, "vs_sourcegate": d_sg}
    r = out["gates"]["unanimous_free_signals"]
    print(f"{'ALL free signals unanimous':<30}{unan.mean():>7.1%}"
          f"{r['routing_acc_agree']:>8.3f}{r['routing_acc_disagree']:>9.3f}"
          f"{gate.mean():>9.4f}{d_union['mean']:>+11.4f}"
          f"{d_union['p_two_sided']:>8.3f}{(~unan).mean():>8.1%}")

json.dump(out, open(RESULTS / "cheap_second_opinion.json", "w"), indent=2)
print("\n  C1 cheap proxies match the reader -> mechanism is ensemble disagreement")
print("  C2 reader gates best              -> reader sees what embeddings cannot")
print(f"\nSaved -> {RESULTS / 'cheap_second_opinion.json'}")
