"""
plot_paper_figures.py

Generates all publication-quality figures for:
"Differentiable Source Routing for Multi-Source RAG"

Figures produced:
  fig1_main_results.pdf         — recall@picked + macro across all methods (main table)
  fig2_per_type_routing_k3.pdf  — per-type routing accuracy K=3 (text/table/kg)
  fig3_per_type_routing_k5.pdf  — per-type routing accuracy K=5 (5 sources)
  fig4_training_curves_k3.pdf   — Phase 3 K=3 val macro over epochs (3 seeds)
  fig5_phase4_loss_curves.pdf   — Phase 4 training loss components over steps
  fig6_k3_vs_k5.pdf             — K=3 vs K=5 macro lift comparison
  fig7_f1_by_source.pdf         — F1 by source type showing LLM limitation
  fig8_gradient_bridge.pdf      — Gradient bridge verification bar chart
  fig9_recall_oracle_gap.pdf    — recall@picked showing headroom to oracle

Run from the mmrag_benchmark directory:
  python plot_paper_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =============================================================================
# Style — clean, IEEE/ACL compatible
# =============================================================================
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

# Colour palette — consistent across all figures
C = {
    "random":    "#b0b0b0",
    "majority":  "#888888",
    "oracle":    "#2ca02c",
    "no_routing":"#ff7f0e",
    "phase3_k3": "#1f77b4",
    "phase4_k3": "#d62728",
    "phase3_k5": "#9467bd",
    "phase4_k5": "#8c564b",
    "text":      "#4c72b0",
    "table":     "#dd8452",
    "kg":        "#55a868",
    "nq":        "#4c72b0",
    "triviaqa":  "#64b5cd",
    "ott":       "#dd8452",
    "tat":       "#e8936a",
    "bridge":    "#d62728",
}


# =============================================================================
# Load data
# =============================================================================
def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        print(f"  [warn] {path} not found, using default")
        return default
    with open(p) as f:
        return json.load(f)


print("Loading result files...")
k3_summary   = load_json("k3_summary.json", {})
k5_summary   = load_json("k5_summary.json", {})
k3_log       = load_json("routing_pretrain_k3_log.json", {})
k5_log       = load_json("routing_pretrain_k5_log.json", {})
p5_metrics   = load_json("phase5_results/phase5_metrics.json", {})

# Phase 4 seed logs
p4_logs = {}
for seed in [42, 123, 2026]:
    log = load_json(f"phase4_logs/phase4_seed{seed}_log.json")
    if log:
        p4_logs[seed] = log
k5_p4_logs = {}
for seed in [7, 99, 314]:
    log = load_json(f"phase4_logs_k5/phase4_k5_seed{seed}_log.json")
    if log:
        k5_p4_logs[seed] = log


# =============================================================================
# Helper: save figure
# =============================================================================
def save(fig, name):
    path = OUT_DIR / f"{name}.pdf"
    fig.savefig(path)
    path2 = OUT_DIR / f"{name}.png"
    fig.savefig(path2)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Fig 1: Main results — recall@picked + routing macro
# =============================================================================
def fig1_main_results():
    print("Plotting Fig 1: Main results...")

    # Shortened labels (single line where possible) -- the original
    # two-line "Superv.\nPretr. K=3" style, combined with 9 categories
    # (up from 8 pre-BGE-conf), overlapped badly at the previous figure
    # width. Widened the figure and shortened labels together for a
    # robust fix rather than just one or the other.
    methods = ["Random", "Majority", "No-routing", "Oracle", "BGE-conf",
               "Superv.\nK=3", "Joint\nK=3",
               "Superv.\nK=5", "Joint\nK=5"]
    colours = [C["random"], C["majority"], C["no_routing"], C["oracle"], "#9467bd",
               C["phase3_k3"], C["phase4_k3"],
               C["phase3_k5"], C["phase4_k5"]]

    # BGE-conf recall corrected 0.778 -> 0.755: it had been computed by the
    # loose path (top-10 per constituent index, i.e. up to 20 chunks for
    # text/table) while SourceGate/oracle used the canonical merged global
    # top-10. See script_32_recall_canonical.py.
    recall = [0.385, 0.558, None, 0.802, 0.755, 0.714, 0.725, None, 0.717]
    # macro routing accuracy -- ALL bars are test-set values, matching the
    # figure's own suptitle and panel (a). Previously the four SourceGate
    # bars carried dev-set macros (0.735/0.709/0.707/0.691, Table E.1)
    # while Random/Majority/Oracle/BGE-conf carried test-set macros, which
    # made the panel a dev/test mixture that the caption had to disclaim.
    # Now every bar is the test-set figure from Table 4:
    #   BGE-conf 0.661 | SG K=3 0.737 / 0.717 | SG K=5 0.678 / 0.697
    # K=5 supervised is 0.678 +- 0.005: the value previously carried here,
    # 0.706 +- 0.012, was the DEVELOPMENT macro mislabelled as test (see
    # k5_seed_extension.json, which gives dev 0.707 +- 0.005 against test
    # 0.678 +- 0.005 over seeds 7/99/314). K=5 joint is 0.697 +- 0.011.
    # Dev-set macros remain available in Table E.1.
    #
    # The two K=5 configurations come from SEPARATE evaluation passes and are
    # not comparable to each other (Table 4, footnote flat). Each therefore
    # appears only on the metric its own pass produced -- K=5 supervised on
    # macro, K=5 joint on recall@picked -- so the panels never juxtapose them.
    macro  = [0.337, 0.333, None, 1.000, 0.661, 0.737, 0.717, 0.678, None]

    x = np.arange(len(methods))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=False)

    for ax, vals, ylabel, title in zip(
        axes,
        [recall, macro],
        ["Recall@Picked", "Routing Macro Accuracy"],
        ["(a) Retrieval Quality: Recall@Picked", "(b) Routing Accuracy: Macro"],
    ):
        bars = []
        for i, (v, c) in enumerate(zip(vals, colours)):
            if v is None:
                bars.append(ax.bar(x[i], 0, w, color=c, alpha=0.3, hatch="//"))
                ax.text(x[i], 0.05, "N/A", ha="center", va="bottom",
                        fontsize=8, color="gray", rotation=90)
            else:
                b = ax.bar(x[i], v, w, color=c, alpha=0.88, edgecolor="white", linewidth=0.5)
                bars.append(b)
                ax.text(x[i], v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.12)

        # Highlight model methods (indices shifted by BGE-conf insertion at 4)
        for i in [5, 6, 7, 8]:
            ax.get_xticklabels()[i].set_color(C["phase3_k3"] if i in [5, 7] else C["phase4_k3"])

    # Dashed line at oracle recall
    axes[0].axhline(0.802, color=C["oracle"], linestyle="--", linewidth=1,
                    alpha=0.5, label="Oracle ceiling")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].axhline(1.000, color=C["oracle"], linestyle="--", linewidth=1, alpha=0.5)

    fig.suptitle("SourceGate Results on mmRAG Test Set", fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig1_main_results")


# =============================================================================
# Fig 2: Per-type routing accuracy K=3
# =============================================================================
def fig2_per_type_k3():
    print("Plotting Fig 2: Per-type routing accuracy K=3...")

    # From Phase 3 K=3 best checkpoints (mean ± std across seeds)
    # From k3_summary.json per_type_mean
    pt = k3_summary.get("per_type_mean", {})
    types_k3 = ["text", "table", "kg"]
    labels = ["Text\n(NQ+TriviaQA)", "Table\n(OTT+TAT)", "KG\n(Freebase)"]

    p3_means = [pt.get(t, {}).get("mean", 0) for t in types_k3]
    p3_stds  = [pt.get(t, {}).get("std",  0) for t in types_k3]

    # Phase 4 per-type from seed logs
    p4_per_type = {t: [] for t in types_k3}
    for seed, log in p4_logs.items():
        final_pt = log.get("final", {}).get("per_type_acc", {})
        for t in types_k3:
            if t in final_pt:
                p4_per_type[t].append(final_pt[t])
    p4_means = [np.mean(p4_per_type[t]) if p4_per_type[t] else 0 for t in types_k3]
    p4_stds  = [np.std(p4_per_type[t])  if p4_per_type[t] else 0 for t in types_k3]

    x = np.arange(len(types_k3))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, p3_means, w, yerr=p3_stds, capsize=4,
                color=C["phase3_k3"], alpha=0.85, label="Supervised pretraining", edgecolor="white")
    b2 = ax.bar(x + w/2, p4_means, w, yerr=p4_stds, capsize=4,
                color=C["phase4_k3"], alpha=0.85, label="Joint training", edgecolor="white")

    # Offset each value label above the ERROR BAR CAP, not the bar top.
    # Placing it at h + 0.02 put the text inside the whisker, which clipped
    # digits -- "0.68" read as "0.58" and "0.72" as "0.?2".
    for bars, errs in ((b1, p3_stds), (b2, p4_stds)):
        for bar, e in zip(bars, errs):
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + (e or 0) + 0.025,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(1/3, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Random (1/K)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Per-Type Routing Accuracy")
    ax.set_title("K=3: Per-Source-Type Routing Accuracy\n(Supervised Pretraining vs Joint Training)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    plt.tight_layout()
    save(fig, "fig2_per_type_routing_k3")


# =============================================================================
# Fig 3: Per-type routing accuracy K=5
# =============================================================================
def fig3_per_type_k5():
    print("Plotting Fig 3: Per-type routing accuracy K=5...")

    types_k5   = ["nq", "triviaqa", "ott", "tat", "kg"]
    labels_k5  = ["NQ", "TriviaQA", "OTT-QA", "TAT-QA", "KG"]
    type_cols  = [C["nq"], C["triviaqa"], C["ott"], C["tat"], C["kg"]]

    # Phase 3 K=5 per-type, read from the three paper checkpoints' own
    # val_per_type at their best-macro epoch (seeds 7/99/314, matching
    # Table E.1's "computed at the best-macro checkpoint").
    #
    # NOT from routing_pretrain_k5_log.json: that file was overwritten on
    # 2026-09-25 by the ten-seed extension runs and now holds seeds
    # 31337/8675309/271828/161803, none of which are the paper's K=5 seeds.
    # Reading it gave a 4-seed mean (nq 0.59, triviaqa 0.49, ott 0.75,
    # tat 0.94, kg 0.77) whose macro, 0.7103, does not match the published
    # 0.707 +- 0.005. The checkpoints below reconstruct it exactly.
    import torch
    p3_pt = {t: [] for t in types_k5}
    for seed in (7, 99, 314):
        ck = torch.load(f"k5_n3_backup/sourceformer_k5_seed{seed}_best.pt",
                        map_location="cpu", weights_only=False)
        vpt = ck["val_per_type"]
        for t in types_k5:
            if t in vpt:
                p3_pt[t].append(float(vpt[t]))
    p3_means = [np.mean(p3_pt.get(t, [0])) for t in types_k5]
    p3_stds  = [np.std(p3_pt.get(t, [0]))  for t in types_k5]

    # Phase 4 K=5 per-type from final val_metrics
    p4_pt = {t: [] for t in types_k5}
    for seed, log in k5_p4_logs.items():
        final_pt = log.get("final", {}).get("per_type_acc", {})
        for t in types_k5:
            if t in final_pt:
                p4_pt[t].append(final_pt[t])
    p4_means = [np.mean(p4_pt[t]) if p4_pt[t] else 0 for t in types_k5]
    p4_stds  = [np.std(p4_pt[t])  if p4_pt[t] else 0 for t in types_k5]

    x = np.arange(len(types_k5))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b1 = ax.bar(x - w/2, p3_means, w, yerr=p3_stds, capsize=4,
                color=C["phase3_k5"], alpha=0.85, label="Supervised pretraining", edgecolor="white")
    b2 = ax.bar(x + w/2, p4_means, w, yerr=p4_stds, capsize=4,
                color=C["phase4_k5"], alpha=0.85, label="Joint training", edgecolor="white")

    # Offset each value label above the ERROR BAR CAP, not the bar top.
    # Placing it at h + 0.02 put the text inside the whisker, which clipped
    # digits -- "0.68" read as "0.58" and "0.72" as "0.?2".
    for bars, errs in ((b1, p3_stds), (b2, p4_stds)):
        for bar, e in zip(bars, errs):
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + (e or 0) + 0.025,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0.2, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Random (1/K)")

    # Shade NQ/TriviaQA as "within-text" difficulty
    ax.axvspan(-0.5, 1.5, alpha=0.04, color="blue", label="Within-modality (harder)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_k5, fontsize=10)
    ax.set_ylabel("Per-Type Routing Accuracy")
    ax.set_title("K=5: Per-Dataset Routing Accuracy\n(Supervised Pretraining vs Joint Training)", fontweight="bold")
    # Headroom above the tallest bar (TAT-QA ~0.94) so the legend does not
    # sit on top of its value label; legend moved out of the plot area
    # entirely, and the within-modality note lowered clear of the NQ/TriviaQA
    # value labels. Both previously collided.
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, fontsize=8.5, frameon=False)

    # Annotation for within-modality difficulty
    ax.annotate("Within-modality\n(harder to distinguish)",
                xy=(0.5, 0.18), fontsize=8.5, color="steelblue",
                ha="center", style="italic")

    plt.tight_layout()
    save(fig, "fig3_per_type_routing_k5")


# =============================================================================
# Fig 4: Phase 3 K=3 training curves (val macro over epochs)
# =============================================================================
def fig4_training_curves_k3():
    print("Plotting Fig 4: Phase 3 K=3 training curves...")

    seeds = [42, 123, 2026]
    seed_colours = [C["phase3_k3"], C["phase4_k3"], C["phase3_k5"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax_macro, ax_type = axes

    for seed, col in zip(seeds, seed_colours):
        seed_key = f"seed_{seed}"
        data = k3_log.get(seed_key, {})
        epochs_data = data.get("epochs", [])
        if not epochs_data:
            continue
        epochs   = [e["epoch"] for e in epochs_data]
        val_macro = [e.get("val_macro", 0) for e in epochs_data]
        val_acc   = [e.get("val_acc", 0)   for e in epochs_data]
        ax_macro.plot(epochs, val_macro, color=col, linewidth=2,
                      marker="o", markersize=3, label=f"Seed {seed}")

    # Majority macro baseline
    ax_macro.axhline(0.333, color="gray", linestyle="--", linewidth=1,
                     alpha=0.7, label="Majority baseline")
    ax_macro.set_xlabel("Epoch")
    ax_macro.set_ylabel("Val Macro Accuracy")
    ax_macro.set_title("(a) K=3 Pretraining: Val Macro", fontweight="bold")
    ax_macro.legend(fontsize=9)
    ax_macro.set_ylim(0.2, 0.82)

    # Per-type at best epoch for best seed
    best_seed_key = max(k3_log.keys(),
                        key=lambda k: k3_log[k].get("best_val_macro", 0)) if k3_log else None
    if best_seed_key:
        epochs_data = k3_log[best_seed_key].get("epochs", [])
        for t, col in zip(["text", "table", "kg"],
                          [C["text"], C["table"], C["kg"]]):
            vals = [e.get("val_per_type", {}).get(t, float("nan"))
                    for e in epochs_data]
            epochs = [e["epoch"] for e in epochs_data]
            ax_type.plot(epochs, vals, color=col, linewidth=2,
                         marker="o", markersize=3, label=t.capitalize())
        ax_type.axhline(0.333, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax_type.set_xlabel("Epoch")
        ax_type.set_ylabel("Per-Type Accuracy")
        ax_type.set_title(f"(b) K=3 Per-Type ({best_seed_key.replace('_', ' ')})", fontweight="bold")
        ax_type.legend(fontsize=9)
        ax_type.set_ylim(0, 1.05)

    plt.tight_layout()
    save(fig, "fig4_training_curves_k3")


# =============================================================================
# Fig 5: Phase 4 training loss curves
# =============================================================================
def fig5_phase4_loss_curves():
    print("Plotting Fig 5: Phase 4 training loss curves...")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    seed_colours = {42: C["phase3_k3"], 123: C["phase4_k3"], 2026: C["phase3_k5"]}

    for ax, loss_key, title in zip(
        axes,
        ["l_route_raw", "l_ans_raw", "l_total"],
        ["L_route (KL Routing Loss)", "L_ans (NLL Answer Loss)", "L_total (Combined)"],
    ):
        for seed, log in p4_logs.items():
            steps_data = log.get("steps", [])
            if not steps_data:
                continue
            steps = [s["opt_step"] for s in steps_data]
            vals  = [s.get(loss_key, 0) for s in steps_data]
            # Smooth with rolling mean (window=5)
            if len(vals) > 5:
                vals_smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
                steps_smooth = steps[2:-2]
            else:
                vals_smooth, steps_smooth = vals, steps
            col = seed_colours.get(seed, "gray")
            ax.plot(steps_smooth, vals_smooth, color=col, linewidth=1.5,
                    alpha=0.85, label=f"Seed {seed}")

        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Loss Value")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8.5)

    plt.suptitle("Joint Training K=3 Loss Curves (smoothed)", fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig5_phase4_loss_curves")


# =============================================================================
# Fig 6: K=3 vs K=5 comparison
# =============================================================================
def fig6_k3_vs_k5():
    print("Plotting Fig 6: K=3 vs K=5 comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))

    categories = ["Supervised\nPretraining", "Joint\nTraining"]
    x = np.arange(len(categories))
    w = 0.35

    # Macro
    # Matches Table C.13 exactly (dev-set, 3-seed mean +- std)
    k3_macros = [0.735, 0.709]
    k5_macros = [0.707, 0.691]
    k3_macro_stds = [0.006, 0.009]
    k5_macro_stds = [0.005, 0.009]

    # Lift over majority
    k3_lift = [0.735-0.333, 0.709-0.333]
    k5_lift = [0.707-0.200, 0.691-0.200]

    # Recall@picked
    k3_recall = [0.714, 0.725]
    k5_recall = [None, 0.717]

    # NLL
    k3_nll = [3.782, 3.818]
    k5_nll = [None, 3.790]

    for ax, (k3_vals, k5_vals, k3_std, k5_std, ylabel, title) in zip(axes, [
        (k3_macros, k5_macros, k3_macro_stds, k5_macro_stds,
         "Macro Accuracy", "(a) Routing Macro Accuracy (dev)"),
        (k3_lift, k5_lift, [0]*2, [0]*2,
         "Macro Lift over Majority", "(b) Lift over Majority Baseline (dev)"),
        (k3_recall, k5_recall, [0]*2, [0]*2,
         "Recall@Picked", "(c) Recall@Picked (test)"),
    ]):
        # A None entry (e.g. K=5 supervised-pretraining recall@picked, which
        # is genuinely "---" / not measured in Table 4) previously fell
        # through to `v if v is not None else 0`, plotting a real-looking
        # zero-height bar indistinguishable from an actual zero measurement.
        # Give it the same hatched-bar + "N/A" label treatment already used
        # in fig1_main_results() for the same situation, rather than
        # inventing a zero that was never measured.
        k3_plot = [v if v is not None else 0 for v in k3_vals]
        k5_plot = [v if v is not None else 0 for v in k5_vals]
        b1 = ax.bar(x - w/2, k3_plot, w, yerr=k3_std, capsize=4,
                    color=C["phase3_k3"], alpha=0.85, label="K=3", edgecolor="white")
        b2 = ax.bar(x + w/2, k5_plot, w, yerr=k5_std, capsize=4,
                    color=C["phase3_k5"], alpha=0.85, label="K=5", edgecolor="white")
        for bars, orig_vals in [(b1, k3_vals), (b2, k5_vals)]:
            for bar, orig_v in zip(bars, orig_vals):
                if orig_v is None:
                    bar.set_alpha(0.3)
                    bar.set_hatch("//")
                    ax.text(bar.get_x() + bar.get_width()/2, 0.01, "N/A",
                            ha="center", va="bottom", fontsize=7.5,
                            color="gray", rotation=90)
                    continue
                h = bar.get_height()
                if h > 0.005:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=10)
        # Legend outside the axes and headroom above the tallest bar: with
        # loc="best" matplotlib put it on top of the 0.507 value label in
        # panel (b) and the "N/A" marker in panel (c).
        top = max([v for v in (k3_vals + k5_vals) if v is not None] or [1.0])
        ax.set_ylim(0, top * 1.30)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, frameon=False)

    plt.suptitle("K=3 vs K=5 Routing Comparison", fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig6_k3_vs_k5")


# =============================================================================
# Fig 7: F1 by source type — LLM limitation analysis
# =============================================================================
def fig7_f1_by_source():
    print("Plotting Fig 7: F1 by source type...")

    # Corrected-pipeline per-type F1, computed from reader_scaling_llama8b_
    # predictions_{oracle,phase3_seed42,phase4_seed42,random}.jsonl, grouped
    # by oracle_type. Majority is excluded: no corrected-pipeline prediction
    # file exists for it yet (only the pre-fix predictions_majority.jsonl),
    # so it is left out rather than mixed with stale data.
    methods = ["Random", "Oracle", "Superv.\nPretr. K=3", "Joint\nTrain. K=3"]
    text_f1  = [0.357, 0.458, 0.438, 0.438]
    table_f1 = [0.092, 0.206, 0.182, 0.189]
    kg_f1    = [0.222, 0.311, 0.280, 0.244]

    x = np.arange(len(methods))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.8))

    b1 = ax.bar(x - w,   text_f1,  w, color=C["text"],  alpha=0.85, label="Text (NQ+TriviaQA)", edgecolor="white")
    b2 = ax.bar(x,       table_f1, w, color=C["table"], alpha=0.85, label="Table (OTT+TAT)", edgecolor="white")
    b3 = ax.bar(x + w,   kg_f1,    w, color=C["kg"],    alpha=0.85, label="KG (Freebase)", edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("End-to-End F1 by Source Type\n(Table QA remains the hardest source for Llama 3.1 8B)",
                 fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax.set_ylim(0, 0.62)

    # Annotation for table ceiling (Oracle's OTT vs TAT split, corrected pipeline)
    # -- placed in the empty upper-left region (no bar there reaches above
    # ~0.46) rather than over the taller Superv./Joint bars on the right.
    oracle_idx = methods.index("Oracle")
    ax.annotate("OTT: retrieval failure (F1=0.120)\nTAT: arithmetic failure (F1=0.304)",
                xy=(oracle_idx, table_f1[oracle_idx]), xytext=(-0.3, 0.53),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=8.5, color="gray", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    save(fig, "fig7_f1_by_source")


# =============================================================================
# Fig 8: Gradient bridge verification
# =============================================================================
def fig8_gradient_bridge():
    print("Plotting Fig 8: Gradient bridge verification...")

    # NOTE: k3_summary.json's "total_grad_norm" field stores a different,
    # mis-scaled quantity than the true summed gradient norm reported in the
    # manuscript text (Section 6.8) -- this was the exact mismatch flagged
    # by Reviewer 2 (text: 11.08/12.19, prior figure: 0.0008/0.0031/0.0034).
    # Using the confirmed-correct values directly rather than that field.
    #
    # A prior version of this figure had a third bar ("Combined", alpha=0.5)
    # hardcoded to the identical value as "L_ans only" (11.0800 / 12.1900
    # repeated to 4 decimals) -- there was never an independent alpha=0.5
    # gradient-norm measurement behind that bar; it was a copy-paste
    # placeholder. Section 6.8 of the manuscript reports exactly two
    # measured conditions -- bridge-absent (Lroute-only, alpha=1.0) and
    # bridge-active (alpha=0.5, "the full answer-quality loss") -- so this
    # figure now plots only those two, matching the text and Table 7
    # exactly, rather than inventing a third data point with no
    # measurement behind it.
    labels = ["L_route only\n(bridge absent, α=1.0)", "Bridge active\n(α=0.5)"]

    gnorms_k3 = [0.0008, 11.08]
    gnorms_k5 = [0.0007, 12.19]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    b1 = ax.bar(x - w/2, gnorms_k3, w, color=C["phase4_k3"], alpha=0.85,
                label="K=3", edgecolor="white")
    b2 = ax.bar(x + w/2, gnorms_k5, w, color=C["phase4_k5"], alpha=0.85,
                label="K=5", edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.00005,
                        f"{h:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Gradient norm (log scale)")
    ax.set_title("Gradient Bridge Verification\n(gradient from L_ans flows through Gumbel-STE)",
                 fontweight="bold")
    ax.legend()

    # Annotate the bridge contribution
    ax.annotate("Bridge active:\nL_ans gradient\nreaches SourceGate",
                xy=(1 + w/2, 12.19),
                xytext=(0.9, 2.0),
                arrowprops=dict(arrowstyle="->", color=C["bridge"]),
                fontsize=8.5, color=C["bridge"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", alpha=0.8))

    plt.tight_layout()
    save(fig, "fig8_gradient_bridge")


# =============================================================================
# Fig 9: recall@picked — showing headroom to oracle
# =============================================================================
def fig9_recall_oracle_gap():
    print("Plotting Fig 9: Recall@picked with oracle gap...")

    methods  = ["Random", "Majority", "BGE-conf", "Superv.\nPretr. K=3", "Joint\nTrain. K=3", "Joint\nTrain. K=5", "Oracle"]
    recalls  = [0.385,    0.558,      0.755,       0.714,               0.725,             0.717,             0.802]
    colours  = [C["random"], C["majority"], "#9467bd", C["phase3_k3"],
                C["phase4_k3"], C["phase4_k5"], C["oracle"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    bars = ax.bar(range(len(methods)), recalls, color=colours, alpha=0.88,
                  edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, recalls)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    # Oracle line and gap arrows
    oracle_val = 0.802
    ax.axhline(oracle_val, color=C["oracle"], linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Oracle ceiling ({oracle_val:.3f})")

    # Gap annotation for SourceGate's own headroom (Joint Training K=3, index 4)
    # -- kept on SourceGate rather than BGE-confidence to match the unchanged
    # manuscript text (Section 6.1 and this figure's caption both cite the
    # 0.077 SourceGate-to-oracle gap as "remaining headroom for retrieval-stage
    # improvements", a claim specifically about SourceGate's own system, not
    # about which method is closest to oracle overall). BGE-confidence's bar
    # is still shown in the chart; it is just not the annotated gap.
    # The arrow sits in the gap between the Joint K=3 and Joint K=5 bars (x=4.5)
    # and starts above the "0.725" bar label, so that neither the arrow nor its
    # caption collides with the 0.725 / 0.717 value labels. The caption is placed
    # above the oracle ceiling line, where the axes are empty at this x position.
    sg_val = 0.725
    # short dotted leader from the Joint K=3 bar top so the gap is unambiguously
    # read against 0.725 rather than against the neighbouring K=5 bar
    ax.plot([4.38, 4.62], [sg_val, sg_val], color="gray", ls=":", lw=1.0,
            alpha=0.9, zorder=3)
    ax.annotate("", xy=(4.5, oracle_val), xytext=(4.5, sg_val),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(4.5, oracle_val + 0.030,
            f"Δ={oracle_val-sg_val:.3f} (headroom)",
            fontsize=8.5, color="gray", ha="center", va="bottom")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Recall@Picked (top-10)")
    ax.set_title("Retrieval Quality: Recall@Picked by Routing Method\n"
                 "Fraction of queries where top-10 chunks contain a relevant answer",
                 fontweight="bold")
    ax.set_ylim(0.25, 0.92)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save(fig, "fig9_recall_oracle_gap")


# =============================================================================
# Fig 10: Phase 4 val macro trajectory (full dev) — K=3 across seeds
# =============================================================================
def fig10_phase4_val_trajectory():
    print("Plotting Fig 10: Phase 4 val trajectory...")

    # NOTE: "validations" entries are cheap periodic monitoring checkpoints
    # computed on a small subset of dev queries (confirmed: step-200 text
    # accuracy of 0.35294117647058826 = 6/17, far too small a fraction for
    # the ~372-query full dev text split) -- NOT the full dev set, despite
    # the axis label previously claiming so. This is exactly the mismatch
    # Reviewer 2 flagged: the trajectory's endpoint (~0.60-0.62 macro)
    # never matched Table 7 / Table C.12's reported values (~0.70-0.72),
    # because those tables report "final" -- the true full-dev-set
    # evaluation at the end of training -- which the trajectory never
    # plotted. Fixed by appending "final" as the true endpoint, visually
    # distinguished (star marker) from the sampled intermediate points,
    # and correcting the axis labels and caption accordingly.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    seed_colours = {42: C["phase3_k3"], 123: C["phase4_k3"], 2026: C["phase3_k5"]}

    ax_macro, ax_recall = axes

    for seed, log in p4_logs.items():
        vals  = log.get("validations", [])
        pre   = log.get("pre", {})
        final = log.get("final", {})
        if not vals:
            continue
        col = seed_colours.get(seed, "gray")
        # Final point placed one step past the last training step, so it's
        # visually separated from the (small-subset) monitoring trajectory.
        final_step = max([s["opt_step"] for s in log.get("steps", [])] +
                          [v["opt_step"] for v in vals]) + 40

        # Sampled-subset monitoring trajectory (steps 0..1200)
        steps   = [0] + [v["opt_step"] for v in vals]
        macros  = [pre.get("macro_acc", 0)] + [v["macro_acc"] for v in vals]
        recalls = [pre.get("recall_at_picked_source", 0)] + [v["recall_at_picked_source"] for v in vals]

        ax_macro.plot(steps, macros, color=col, linewidth=2, marker="o",
                      markersize=5, alpha=0.55, label=f"Seed {seed} (monitoring subset)")
        ax_recall.plot(steps, recalls, color=col, linewidth=2, marker="s",
                       markersize=5, alpha=0.55, label=f"Seed {seed} (monitoring subset)")

        # True full-dev endpoint, matching Table 7 / Table C.12
        if final:
            ax_macro.plot([steps[-1], final_step],
                          [macros[-1], final.get("macro_acc", macros[-1])],
                          color=col, linewidth=1, linestyle=":", alpha=0.7)
            ax_macro.plot(final_step, final.get("macro_acc", None),
                         color=col, marker="*", markersize=14,
                         markeredgecolor="black", markeredgewidth=0.6, zorder=5)
            ax_recall.plot([steps[-1], final_step],
                           [recalls[-1], final.get("recall_at_picked_source", recalls[-1])],
                           color=col, linewidth=1, linestyle=":", alpha=0.7)
            ax_recall.plot(final_step, final.get("recall_at_picked_source", None),
                          color=col, marker="*", markersize=14,
                          markeredgecolor="black", markeredgewidth=0.6, zorder=5)

    for ax, ylabel, title in zip(
        [ax_macro, ax_recall],
        ["Val Macro Accuracy", "Val Recall@Picked"],
        ["(a) Routing Macro During Joint Training", "(b) Recall@Picked During Joint Training"],
    ):
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.text(10, ax.get_ylim()[0] + 0.01, "← Supervised\npretraining\ncheckpoint",
                fontsize=7, color="gray")

    fig.text(0.5, -0.04,
              "Circles/squares: periodic monitoring on a small dev subset (noisier, for training diagnostics only). "
              "Star: true full-dev-set evaluation, matching Table 10.",
              ha="center", fontsize=8, style="italic", color="#444444")

    plt.suptitle("Joint Training K=3 Progress", fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig10_phase4_val_trajectory")


# =============================================================================
# Run all
# =============================================================================
if __name__ == "__main__":
    print(f"\nGenerating paper figures → {OUT_DIR}/\n")
    fig1_main_results()
    fig2_per_type_k3()
    fig3_per_type_k5()
    fig4_training_curves_k3()
    fig5_phase4_loss_curves()
    fig6_k3_vs_k5()
    fig7_f1_by_source()
    fig8_gradient_bridge()
    fig9_recall_oracle_gap()
    fig10_phase4_val_trajectory()

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("\nFigure inventory:")
    for f in sorted(OUT_DIR.glob("*.pdf")):
        print(f"  {f.name}")


# =============================================================================
# Fig 12: macro-accuracy / recall@picked Pareto frontier
# =============================================================================
def fig12_pareto_frontier():
    """Scatter of the two routing metrics with the Pareto frontier marked.

    Oracle is deliberately excluded: it is a ceiling (macro 1.000 by
    construction), not a competing method, and including it would trivially
    dominate every point. If the recall-aware sweep from
    script_30_recall_aware_routing.py has been run, its test curve is
    overlaid to show SourceGate's reachable operating points.
    """
    print("Plotting Fig 12: Pareto frontier...")

    pts = {  # name: (routing macro, recall@picked)  -- Table 4, test set
        "Random":        (0.337, 0.385), "Majority":      (0.333, 0.558),
        # PrefRAG-Conf and R^3AG-full corrected for the chunk-lookup defect:
        # (0.334, 0.299) -> (0.588, 0.659) and (0.348, 0.347) -> (0.482, 0.536).
        # Logistic regression corrected for the mixed-computation defect:
        # (0.642, 0.682) -> (0.644, 0.691). All three remain dominated by
        # SourceGate on both axes, so the frontier is unchanged.
        "PrefRAG-Conf":  (0.588, 0.659), "R$^3$AG-full":  (0.482, 0.536),
        "GPT-4o-mini":   (0.542, 0.624), "Logistic Reg.": (0.644, 0.691),
        "BGE-conf":      (0.661, 0.755), "R$^3$AG-RQ":    (0.663, 0.709),
        "MLP-HardCE":    (0.702, 0.709),
        "SourceGate (Joint)":   (0.717, 0.725),
        "SourceGate (Superv.)": (0.737, 0.714),
    }
    sg = {"SourceGate (Joint)", "SourceGate (Superv.)"}

    def dominated(nm):
        m, r = pts[nm]
        return any(m2 >= m and r2 >= r and (m2 > m or r2 > r)
                   for o, (m2, r2) in pts.items() if o != nm)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # Empty dominance quadrant for the SourceGate points: anything strictly
    # better than SourceGate on BOTH metrics would have to sit in here.
    for nm in sg:
        m, r = pts[nm]
        ax.add_patch(plt.Rectangle((m, r), 1.05 - m, 0.83 - r,
                                   facecolor=C["phase3_k3"], alpha=0.07,
                                   edgecolor="none", zorder=0))

    front = sorted([p for p in pts if not dominated(p)], key=lambda n: pts[n][0])
    ax.plot([pts[n][0] for n in front], [pts[n][1] for n in front],
            "--", color="gray", lw=1.2, alpha=0.8, zorder=1,
            label="Pareto frontier")

    for nm, (m, r) in pts.items():
        is_sg, on_f = nm in sg, not dominated(nm)
        ax.scatter(m, r, s=150 if is_sg else 70,
                   marker="*" if is_sg else ("o" if on_f else "x"),
                   color=C["phase3_k3"] if is_sg else ("#9467bd" if on_f else "#b0b0b0"),
                   edgecolor="white" if is_sg else "none",
                   linewidth=1.0, zorder=4)
        # Hand-placed offsets: the five frontier points plus logistic
        # regression sit within ~0.08 of each other on both axes, so a single
        # uniform offset collides. (dx, dy, horizontal alignment)
        off = {
            "BGE-conf":             (0, 10, "center"),
            "R$^3$AG-RQ":           (-9, -3, "right"),
            "MLP-HardCE":           (0, -14, "center"),
            "Logistic Reg.":        (-9, -3, "right"),
            # Right and slightly low: the default right-up offset collided
            # with Logistic Reg., and a below-centre offset collided with
            # GPT-4o-mini.
            "PrefRAG-Conf":         (10, -5, "left"),
            "SourceGate (Joint)":   (9, 4, "left"),
            "SourceGate (Superv.)": (9, -10, "left"),
            "R$^3$AG-full":         (8, -9, "left"),
        }.get(nm, (8, 3, "left"))
        ax.annotate(nm, (m, r), textcoords="offset points",
                    xytext=off[:2], ha=off[2],
                    fontsize=8, color="black" if (is_sg or on_f) else "gray",
                    fontweight="bold" if is_sg else "normal")

    ax.set_xlabel("Routing macro accuracy")
    ax.set_ylabel("Recall@Picked (top-10)")
    ax.set_title("Routing accuracy vs. retrieval hit-rate\n"
                 "(oracle excluded: a ceiling, not a method)",
                 fontweight="bold")
    ax.set_xlim(0.30, 0.90)
    ax.set_ylim(0.27, 0.84)
    ax.legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    save(fig, "fig12_pareto_frontier")
