# SourceGate — code and artifacts

Supervised routing for multi-source retrieval-augmented generation.
Anonymous release accompanying manuscript NEUCOM-D-26-14977 (Neurocomputing, under review).

Everything reported in the paper is reproducible from this bundle, except the FAISS
indices, which are too large to host here and are rebuilt by a script (see below).

---

## Contents

| Directory | What it holds |
|---|---|
| `scripts/` | All training, evaluation and analysis code (92 files), plus the figure generators |
| `results/` | The stored artifacts behind every table and figure (111 files) |
| `checkpoints/` | The 12 router checkpoints behind the reported results: K=3 seeds 42/123/2026 (461,059 parameters) and the K=5 seeds, the paper's 7/99/314 plus the ten-seed extension of Section 7.3 (461,317 parameters). Each carries its own `val_macro` and `val_per_type`. |
| `figures/` | The 13 figures as published |

## The FAISS indices

The five per-source indices total **9.16 GiB** and are not included. Rebuild them with:

```
python scripts/build_text_index.py      # per-source dense indices (BGE-base-en-v1.5)
python scripts/build_chunk_db.py        # chunk-id -> text mapping
```

The mmRAG benchmark itself is public and must be obtained from its original authors;
see the manuscript's Data Availability statement.

## Reproducing the headline results

| Result | Entry point |
|---|---|
| SourceGate training (K=3 and K=5) | `scripts/script_5b_k3_FINAL.py`, `scripts/script_5b_k5.py` |
| Joint training (gradient bridge stage) | `scripts/script_6_joint_training.py` |
| Main results table | `scripts/script_7_evaluation.py`, `scripts/script_10_reader_scaling.py` |
| Confidence baselines (PrefRAG-Conf, BGE-confidence) | `scripts/script_12_prefrag_conf.py`, `scripts/script_25_bge_conf_eval.py` |
| R³AG adaptations | `scripts/script_13_r3ag.py` |
| 2×2 training-signal factorial | `scripts/script_14_ablation_2x2.py` |
| Similarity residual / domination result | `scripts/script_48_additive_residual.py`, `scripts/script_54_subsumption_strength.py` |
| Controlled gradient-bridge arms | `scripts/script_38_bridge_ablation.py` (use `run_bridge_sweep.sh` for the published arm configs) |
| Donor-context format penalty | `scripts/script_40_contrastive_evidence_gain.py`, `scripts/script_57_c1_nuisance_decomposition.py` |
| Mix-versus-selection decomposition | `scripts/script_63_nll_mix_vs_selection.py` |
| Per-query cost benchmark | `scripts/script_29_routing_cost_benchmark.py` |
| All figures | `scripts/plot_paper_figures.py`, `scripts/make_fig_*.py` |

## Notes that will save you time

**Two NLL routines, ~2.4 nats apart.** Every gold-answer NLL in the paper comes from
`compute_l_ans_sequential` (top-10 chunks, 2048-token cap, answer span masked). The
reader-scaling script's `compute_nll` truncates differently and is not used for any
reported value.

**A third quantity, NLL₈₀₀.** The mix-versus-selection decomposition and the
donor-context control are computed on `results/ceg_scores_d10.npz`, a counterfactual
matrix holding scores for *all three* sources of every query under a common
800-character budget. Its absolute values are not comparable with the table column;
the decomposition is internal to the matrix.

**K=5 seeds.** The paper's K=5 results use seeds 7, 99 and 314. Their per-dataset
development accuracies are stored in the checkpoints' own `val_per_type` field —
read them from `checkpoints/sourceformer_k5_seed{7,99,314}_best.pt`, not from a
pretraining log. `results/routing_pretrain_k5_log_THREE_SEEDS.json` is the matching
three-seed log.

**Release FAISS before loading the reader.** Scripts that do both will be OOM-killed
otherwise; `script_40` line 147 shows the pattern.

**Label convention.** The canonical K=3 routing label is the argmax over summed
per-source `dataset_score`, giving 717 text / 359 table / 210 kg on the test split.

**Recall@picked convention.** Per-source-type indices are merged and the global top-10
kept, so each source returns ten chunks in total rather than ten per index.
`results/recall_at_picked_canonical.json` is authoritative.
