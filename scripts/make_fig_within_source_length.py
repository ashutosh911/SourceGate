"""
make_fig_within_source_length.py

R2 asked for a within-source correlation plot of NLL against context length,
stratified by source. The point it settles: if length drove the reader's
score, the relationship would be visible WITHIN each source, not only as a
gap BETWEEN sources. A between-source gap with flat within-source slopes is
a source effect wearing length's clothing.

Uses the contexts actually scored (800-character budget, so token counts vary
by how each source tokenises) and the corrected 10-donor reader scores.

Output: paper_figures/fig_within_source_length.pdf
CPU only.
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = ["text", "table", "kg"]
COL = {"text": "#3b6fb6", "table": "#c1652f", "kg": "#4a8c5c"}
BASE = os.environ.get("SOURCEGATE_ROOT", ".")   # repo root; override if data lives elsewhere
OUTDIR = os.environ.get("SOURCEGATE_FIGS", "figures")
os.makedirs(OUTDIR, exist_ok=True)

z = np.load(f"{BASE}/phase5_results/ceg_scores_d10.npz", allow_pickle=True)
score = z["real"]                      # (Q,3) log-likelihood, higher = better
Q, S = score.shape
ctx = json.load(open(f"{BASE}/phase9_results/retrieved_contexts_all.json"))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokens = np.array([[len(tok(ctx[str(i)][s], add_special_tokens=False)["input_ids"])
                    for s in SRC] for i in range(Q)], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[1.35, 1])

# ---- left: scatter + within-source fit --------------------------------
ax = axes[0]
stats = {}
for j, s in enumerate(SRC):
    x, y = tokens[:, j], score[:, j]
    ax.scatter(x, y, s=5, alpha=0.18, color=COL[s], edgecolors="none")
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color=COL[s], lw=2.2,
            label=f"{s}:  slope {b:+.5f}/tok,  r={np.corrcoef(x, y)[0,1]:+.3f}")
    stats[s] = {"slope_per_token": float(b), "r": float(np.corrcoef(x, y)[0, 1]),
                "mean_tokens": float(x.mean()), "mean_score": float(y.mean())}
ax.set_xlabel("context length actually scored (tokens, at the 800-char budget)")
ax.set_ylabel("reader log-likelihood  (higher = more confident)")
ax.set_title("Within source, length barely moves the score", fontsize=11)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax.grid(alpha=0.25, lw=0.5)

# ---- right: between-source means ---------------------------------------
ax = axes[1]
mt = [stats[s]["mean_tokens"] for s in SRC]
ms = [stats[s]["mean_score"] for s in SRC]
ax.scatter(mt, ms, s=150, c=[COL[s] for s in SRC], zorder=3)
# Margins first, so no marker sits on an axis and no label runs off the edge.
ax.margins(x=0.18, y=0.22)
for s, x, y in zip(SRC, mt, ms):
    # Place each label inside the axes: right-most point labels to its left,
    # the lowest point labels above rather than below.
    right_most = (x == max(mt))
    lowest     = (y == min(ms))
    dx, ha = ((-10, "right") if right_most else (9, "left"))
    dy, va = ((10, "bottom") if lowest else (5, "bottom"))
    ax.annotate(s, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=10, ha=ha, va=va, zorder=4)
ax.set_xlabel("mean context length (tokens)")
ax.set_ylabel("mean reader log-likelihood")
ax.set_title("The gap is between sources, not along length", fontsize=11)
ax.grid(alpha=0.25, lw=0.5)
span_t = max(mt) - min(mt)
span_s = max(ms) - min(ms)
ax.text(0.03, 0.97,
        f"token spread across sources: {span_t:.0f} tok\n"
        f"score spread across sources: {span_s:.3f} nats",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(fc="white", ec="0.7", alpha=0.9))

fig.tight_layout()
out = f"{OUTDIR}/fig_within_source_length.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")

print(f"{'source':8s} {'mean tok':>9} {'slope/tok':>12} {'r':>8} "
      f"{'implied nats over observed range':>34}")
for s in SRC:
    d = stats[s]
    x = tokens[SRC.index(s)]
    rng = tokens[:, SRC.index(s)].max() - tokens[:, SRC.index(s)].min()
    print(f"{s:8s} {d['mean_tokens']:>9.1f} {d['slope_per_token']:>12.6f} "
          f"{d['r']:>8.3f} {d['slope_per_token']*rng:>34.4f}")
print(f"\nbetween-source score spread: {span_s:.4f} nats "
      f"over a {span_t:.0f}-token spread")
with open(f"{BASE}/phase5_results/within_source_length.json", "w") as f:
    json.dump({"per_source": stats, "token_spread": float(span_t),
               "score_spread": float(span_s)}, f, indent=2)
print(f"wrote {out}")
