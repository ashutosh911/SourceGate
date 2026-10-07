"""
make_fig_pareto_cost.py

R2: "no-routing beats you end-to-end and you report no latency/cost. The
confidence baselines need K LLM passes; you need a 461K MLP."

The accuracy comparison alone does not settle that, because the methods do not
cost the same. This plots routing quality against routing cost, so the
trade-off is visible rather than asserted.

IMPORTANT -- reader consistency. Every F1/EM here comes from ONE run of
script_10 with the local Llama-3.1-8B-4bit reader, so the points are mutually
comparable. This is deliberately NOT read from the manuscript's Table 4,
whose F1/EM column appears to have been produced with a different (larger)
reader than its NLL and routing columns; mixing those would make the
trade-off unreadable.

Latencies are the measured routing overheads from Table `tab:cost`
(batch size 1, exact IndexFlatIP on CPU, BGE-base + 461K MLP on RTX 5070 Ti,
Llama 3.1 8B 4-bit). Answer generation is excluded, being identical across
strategies -- except for PrefRAG-Conf, whose reader passes ARE its routing
decision.

CPU only.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = os.environ.get("SOURCEGATE_ROOT", ".")   # repo root; override if data lives elsewhere
OUTDIR = os.environ.get("SOURCEGATE_FIGS", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# routing overhead in ms, from tab:cost
LAT = {
    "majority": 69.1,
    "bge_confidence": 69.2,      # one FAISS search + similarity compare
    "logreg": 69.2,
    "mlp_hardce": 69.2,
    "sourcegate": 69.2,
    "prefrag_conf": 669.5,
    "no_routing": 178.7,
}
NICE = {
    "majority": "Majority", "bge_confidence": "BGE-confidence",
    "logreg": "LogReg", "mlp_hardce": "MLP-HardCE",
    "sourcegate": "SourceGate", "prefrag_conf": "PrefRAG-Conf",
    "no_routing": "No-routing (union)",
}

m = json.load(open(f"{BASE}/phase5_results/reader_scaling_llama8b_metrics.json"))
pts = []
for k, v in m.items():
    if not isinstance(v, dict) or "f1_mean" not in v:
        continue
    if k not in LAT:
        print(f"  [skip] no measured latency for '{k}'")
        continue
    if "routing_macro" not in v:
        # no-routing retrieves the global top-10 over all five indices, so it has
        # no per-source routing decision and no routing macro / recall@picked.
        print(f"  [skip] '{k}' has no per-source routing metrics (union retrieval)")
        continue
    pts.append((NICE.get(k, k), LAT[k], v["routing_macro"], v["f1_mean"],
                v["recall_at_picked"]))

if not pts:
    raise SystemExit("no comparable methods found -- has script_10 finished?")

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
for ax, yi, ylab, title in [
    (axes[0], 2, "routing macro accuracy", "(a) Routing quality vs cost"),
    (axes[1], 3, "end-to-end F1 (Llama-3.1-8B-4bit)", "(b) Answer quality vs cost"),
]:
    for name, lat, macro, f1, rp in pts:
        y = {2: macro, 3: f1}[yi]
        good = name == "SourceGate"
        ax.scatter(lat, y, s=190 if good else 110,
                   c="#c1652f" if good else "#3b6fb6",
                   marker="*" if good else "o", zorder=3,
                   edgecolors="white", linewidths=0.8)
        # Panel (b) packs SourceGate, MLP-HardCE and LogReg into ~0.002 F1 at
        # the same latency, so a single offset stacks the three labels on top
        # of one another. Stagger them vertically there only; panel (a)'s
        # macro values are far enough apart for the default.
        off = (9, 5)
        if yi == 3:
            off = {"SourceGate": (9, 8),
                   "MLP-HardCE": (9, -3),
                   "LogReg":     (9, -14)}.get(name, (9, 5))
        ax.annotate(name, (lat, y), textcoords="offset points",
                    xytext=off, fontsize=8.6)
    ax.set_xscale("log")
    # Minor decade labels (2x10^2, 3x10^2, ...) run together at this width.
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_locator(mticker.FixedLocator([100, 200, 400, 700]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlabel("routing overhead per query (ms, log scale)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5)

fig.tight_layout()
out = f"{OUTDIR}/fig_pareto_cost.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")

print(f"{'method':<22}{'ms':>9}{'macro':>9}{'F1':>9}{'R@picked':>10}")
for name, lat, macro, f1, rp in sorted(pts, key=lambda r: -r[2]):
    print(f"{name:<22}{lat:>9.1f}{macro:>9.4f}{f1:>9.4f}{rp:>10.4f}")
print(f"\nwrote {out}")
