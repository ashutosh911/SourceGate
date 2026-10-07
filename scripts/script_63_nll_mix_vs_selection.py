"""
script_63_nll_mix_vs_selection.py

Is PrefRAG-Conf's near-oracle NLL earned, or an artefact of which sources it
happens to favour?

PrefRAG-Conf reaches gold-answer NLL 3.7887 against oracle's 3.7843 while
routing correctly only 58.8% of the time. Two explanations:

  (A) SELECTION -- it genuinely identifies, per query, the source under which
      the reader will do well.
  (B) MIX -- the three sources have different average NLL (table is hardest),
      and PrefRAG-Conf simply under-picks the hard source. Its per-query
      choices would then carry no information beyond that marginal.

These are separable. Hold each method's marginal pick distribution fixed and
shuffle WHICH queries get which source. Any NLL advantage that survives the
shuffle is mix; any that disappears is genuine per-query selection.

    mix_only(method)  = E[ NLL | picks permuted within the method's marginal ]
    selection_gain    = mix_only - actual      (positive = real selection skill)

Uses ceg_scores_d10.npz, which holds reader scores for ALL THREE sources of
every query -- needed because the shuffle asks "what would this method have
scored had it picked source s for query i". Those are 800-char contexts, so
absolute values differ from the 2048-token NLL column; the decomposition is
internal to this matrix and unaffected.

CONVENTION: ceg `real` is NEGATIVE NLL (higher = better). Reported as NLL
(lower = better) by negation.

CPU only.
"""
import json

import numpy as np

rng = np.random.default_rng(20260930)
SRC = ["text", "table", "kg"]
OUT = "phase5_results/nll_mix_vs_selection.json"
NPERM = 2000

z = np.load("phase5_results/ceg_scores_d10.npz", allow_pickle=True)
nll = -z["real"]                       # (Q,3) NLL, lower is better
gold = z["gold"]
Q = len(gold)

print("per-source mean NLL over ALL queries (the mix effect):")
for j, s in enumerate(SRC):
    print(f"  {s:6s} {nll[:, j].mean():.4f}")

DEC = {
    "oracle":        gold,
    "prefrag_conf":  np.load("phase5_results/prefrag_conf_decisions_k3.npy").astype(int),
    "bge_confidence":np.load("phase5_results/confidence_decisions_k3.npy").astype(int),
    "logreg":        np.load("phase5_results/lr_decisions_k3.npy").astype(int),
    "mlp_hardce":    np.load("phase5_results/mlp_hardce_decisions_k3.npy").astype(int),
    "majority":      np.load("phase5_results/majority_decisions_k3.npy").astype(int),
    "random":        np.random.default_rng(42).integers(0, 3, Q),
}

rows = {}
print(f"\n{'method':<16}{'actual':>9}{'mix-only':>10}{'selection':>11}{'p':>8}{'macro':>8}")
for name, dec in DEC.items():
    dec = np.asarray(dec).astype(int)
    actual = nll[np.arange(Q), dec].mean()
    # permute which query gets which source, holding the marginal fixed
    perm_means = np.empty(NPERM)
    for t in range(NPERM):
        sh = rng.permutation(dec)
        perm_means[t] = nll[np.arange(Q), sh].mean()
    mix = perm_means.mean()
    gain = mix - actual                       # >0 means real per-query skill
    p = float((perm_means <= actual).mean())  # how often chance beats it
    macro = float(np.mean([(dec[gold == c] == c).mean() for c in range(3)]))
    rows[name] = {"actual_nll": float(actual), "mix_only_nll": float(mix),
                  "selection_gain": float(gain), "p_vs_permutation": p,
                  "macro": macro,
                  "pick_dist": {s: int((dec == k).sum()) for k, s in enumerate(SRC)}}
    print(f"{name:<16}{actual:>9.4f}{mix:>10.4f}{gain:>+11.4f}{p:>8.4f}{macro:>8.3f}")

print("\nREAD: 'mix-only' is what this method's SOURCE MIX alone would score.")
print("      'selection' is how much its per-query choices beat that.")
print("      A method whose selection gain is ~0 is riding the marginal.")

o, p_ = rows["oracle"], rows["prefrag_conf"]
print(f"\noracle       selection gain {o['selection_gain']:+.4f}")
print(f"prefrag_conf selection gain {p_['selection_gain']:+.4f}"
      f"   ({100*p_['selection_gain']/o['selection_gain']:.0f}% of oracle's)"
      if o["selection_gain"] else "")
json.dump(rows, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
