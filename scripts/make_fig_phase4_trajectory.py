"""
make_fig_phase4_trajectory.py

Replacement for fig10_phase4_val_trajectory.pdf (Fig A.8).

The problem with the previous version was an asymmetry in its own convention.
Intermediate points are cheap monitoring evaluations on a small dev subset;
the END point was additionally marked with a star showing the true full-dev
value, but the START point was not. A reader therefore compared a
subset-measured start against a full-dev end and saw routing macro climbing
from ~0.28 to ~0.71 -- when the supervised checkpoint's true full-dev macro is
0.7410 and joint training actually *lowers* it to ~0.717. The figure's shape
argued against the paper's own tables. Reviewer 2 flagged this as "caption
says stable while the line doubles".

This version stars BOTH endpoints, so the comparison is like-for-like and the
plotted shape matches Table 7: macro dips slightly, recall rises.

Step-0 full-dev values come from k3_summary.json (per-seed supervised
pretraining, the checkpoint joint training starts from). Full-dev
recall@picked at step 0 was only recorded for seed 42 (0.7298, measured during
the bridge control runs), so only that one is starred on panel (b); the
caption says so rather than inferring the other two.

CPU only.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("SOURCEGATE_ROOT", ".")   # repo root; override if data lives elsewhere
OUTDIR = os.environ.get("SOURCEGATE_FIGS", "figures")
os.makedirs(OUTDIR, exist_ok=True)

COL = {42: "#3b6fb6", 123: "#c1652f", 2026: "#7a5ea8"}
# true full-dev macro of the supervised checkpoint each run starts from
PRE_FULLDEV_MACRO = {r["seed"]: r["macro"]
                     for r in json.load(open(f"{BASE}/k3_summary.json"))["per_seed"]}
# full-dev recall@picked at step 0: only measured for seed 42
PRE_FULLDEV_RECALL = {42: 0.7298}

fig, (ax_m, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.3))

for seed in (42, 123, 2026):
    p = f"{BASE}/phase4_logs/phase4_seed{seed}_log.json"
    if not os.path.exists(p):
        continue
    log = json.load(open(p))
    vals, pre, final = log.get("validations", []), log.get("pre", {}), log.get("final", {})
    if not vals:
        continue
    col = COL[seed]
    steps = [0] + [v["opt_step"] for v in vals]
    macros = [pre.get("macro_acc", 0)] + [v["macro_acc"] for v in vals]
    recalls = [pre.get("recall_at_picked_source", 0)] + [v["recall_at_picked_source"] for v in vals]
    end = max(steps) + 40

    ax_m.plot(steps, macros, color=col, lw=1.8, marker="o", ms=4.5, alpha=0.45,
              label=f"seed {seed} (monitoring subset)")
    ax_r.plot(steps, recalls, color=col, lw=1.8, marker="s", ms=4.5, alpha=0.45,
              label=f"seed {seed} (monitoring subset)")

    # --- the fix: star BOTH ends on the same (full-dev) scale ---
    m0 = PRE_FULLDEV_MACRO.get(seed)
    if m0 is not None and final:
        ax_m.plot([0, end], [m0, final["macro_acc"]], color=col, ls="--", lw=1.6, alpha=0.9)
        ax_m.plot([0, end], [m0, final["macro_acc"]], color=col, marker="*", ms=17,
                  ls="none", markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    r0 = PRE_FULLDEV_RECALL.get(seed)
    if final:
        if r0 is not None:
            ax_r.plot([0, end], [r0, final["recall_at_picked_source"]], color=col,
                      ls="--", lw=1.6, alpha=0.9)
            ax_r.plot([0, end], [r0, final["recall_at_picked_source"]], color=col,
                      marker="*", ms=17, ls="none", markeredgecolor="white",
                      markeredgewidth=0.8, zorder=5)
        else:
            ax_r.plot([end], [final["recall_at_picked_source"]], color=col, marker="*",
                      ms=17, ls="none", markeredgecolor="white", markeredgewidth=0.8, zorder=5)

for ax, lab, title in [(ax_m, "val macro accuracy", "(a) Routing macro: starts high, dips slightly"),
                       (ax_r, "val recall@picked", "(b) Recall@picked: rises")]:
    ax.set_xlabel("optimizer step"); ax.set_ylabel(lab)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, loc="lower right")

ax_m.annotate("true full-dev value\nof the starting checkpoint",
              xy=(0, PRE_FULLDEV_MACRO[42]), xytext=(230, PRE_FULLDEV_MACRO[42] - 0.10),
              fontsize=8.2, color="#444",
              arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))

fig.suptitle("Joint training K=3: full-dev endpoints starred at BOTH ends", fontsize=11.5)
fig.tight_layout()
out = f"{OUTDIR}/fig10_phase4_val_trajectory.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")

print("step-0 full-dev macro (supervised checkpoint):")
for s, v in PRE_FULLDEV_MACRO.items():
    print(f"  seed {s:>5}: {v:.4f}")
print(f"\nwrote {out}")
