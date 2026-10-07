"""
make_fig_bridge_controlled.py

Replacement for fig8_gradient_bridge.pdf.

Why the old figure could not do its job. It compared "L_route only
(alpha=1.0)" against "bridge active (alpha=0.5)" and read the jump
(0.0008 -> 11.08) as evidence the bridge carries gradient. Two problems:

  1. At alpha=1.0 the inner loss carries weight (1 - alpha) = 0, so the
     near-zero gradient is forced by the weighting, not by the bridge being
     absent.
  2. At alpha=0.5 the measured norm includes L_aux as well as L_ans, so the
     jump is not attributable to the bridge.

Both are moot in any case: the published estimator, ste_bridge =
(one_hot.sum(-1)).mean(), is identically 1.0, so its derivative w.r.t. the
router is exactly zero. Whatever produced 11.08 was not the bridge.

This figure instead plots the ANSWER-LOSS-ONLY gradient norm -- obtained by
differentiating L_ans alone w.r.t. the router parameters, with L_route and
L_aux switched off (alpha=0, aux=0) -- across three controlled arms that
differ ONLY in the estimator, alongside what each does to routing. The
zero-gradient arm is the load-bearing control: it must reproduce its starting
point exactly, and does.

CPU only.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("SOURCEGATE_ROOT", ".")   # repo root; override if data lives elsewhere
OUTDIR = os.environ.get("SOURCEGATE_FIGS", "figures")
os.makedirs(OUTDIR, exist_ok=True)

ARMS = [
    ("ansonly_nobridge", "no bridge\n(control)", "#8c8c8c"),
    ("ansonly_baseline", "STE,\nbaselined", "#3b6fb6"),
    ("ansonly_fixed", "STE,\nunbaselined", "#c1652f"),
]

g, pre, fin, lab, col, per = [], [], [], [], [], []
for tag, label, c in ARMS:
    d = json.load(open(f"{BASE}/phase4_logs_bridge_ablation/ba_{tag}_seed42_log.json"))
    g.append(d["answer_bridge_grad_norm"])
    pre.append(d["pre"]["macro_acc"])
    fin.append(d["final"]["macro_acc"])
    per.append(d["final"]["per_type_acc"])
    lab.append(label); col.append(c)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

# --- (a) answer-loss-only gradient norm --------------------------------
ax = axes[0]
gp = [max(v, 1e-3) for v in g]          # log axis needs a floor for the zero bar
b = ax.bar(range(len(g)), gp, color=col, edgecolor="white", width=0.62)
ax.set_yscale("log")
ax.set_ylim(1e-3, 1e2)
for i, v in enumerate(g):
    ax.text(i, gp[i] * 1.25, "0 (exactly)" if v == 0 else f"{v:.2f}",
            ha="center", fontsize=9)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=9)
ax.set_ylabel(r"$\|\partial \mathcal{L}_{\mathrm{ans}} / \partial \theta\|$  (log)")
ax.set_title("(a) Gradient reaching the router\nfrom the answer loss alone", fontsize=10.5)
ax.grid(axis="y", alpha=0.25, lw=0.5)

# --- (b) routing outcome ------------------------------------------------
ax = axes[1]
x = np.arange(len(lab)); w = 0.38
ax.bar(x - w/2, pre, w, color="#cccccc", edgecolor="white", label="before")
ax.bar(x + w/2, fin, w, color=col, edgecolor="white", label="after")
for i in range(len(x)):
    ax.text(x[i] + w/2, fin[i] + 0.012, f"{fin[i]:.4f}", ha="center", fontsize=8.5)
    d = fin[i] - pre[i]
    ax.text(x[i] + w/2, fin[i] - 0.06, f"{d:+.4f}", ha="center", fontsize=8.5,
            color="white" if d < -0.02 else "black")
ax.axhline(pre[0], ls="--", lw=1, color="#555")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9)
ax.set_ylim(0, 0.95); ax.set_ylabel("routing macro accuracy")
ax.set_title("(b) Effect on routing\n(supervised checkpoint start)", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.25, lw=0.5)

# --- (c) per-type, showing the estimator artefact -----------------------
ax = axes[2]
types = ["text", "table", "kg"]
xt = np.arange(len(types)); w = 0.26
for i, (tag, label, c) in enumerate(ARMS):
    vals = [per[i][t] for t in types]
    ax.bar(xt + (i - 1) * w, vals, w, color=c, edgecolor="white",
           label=label.replace("\n", " "))
ax.set_xticks(xt); ax.set_xticklabels(types)
ax.set_ylim(0, 1.0); ax.set_ylabel("per-type routing accuracy")
ax.set_title("(c) The KG concentration is specific\nto the no-baseline estimator", fontsize=10.5)
ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.25, lw=0.5)

fig.tight_layout()
out = f"{OUTDIR}/fig8_bridge_controlled.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")

print(f"{'arm':<26}{'grad':>10}{'PRE':>9}{'FINAL':>9}{'delta':>10}")
for i, (tag, label, _) in enumerate(ARMS):
    print(f"{tag:<26}{g[i]:>10.2f}{pre[i]:>9.4f}{fin[i]:>9.4f}{fin[i]-pre[i]:>+10.4f}")
print(f"\nwrote {out}")
