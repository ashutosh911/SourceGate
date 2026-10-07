#!/usr/bin/env bash
# Assemble the uploaded artifacts into the layout the scripts expect.
# The repository stores them split across several directories; the scripts
# read from phase5_results/ and checkpoints/.
set -euo pipefail

mkdir -p phase5_results checkpoints

cp stage1_results/results-1/*  phase5_results/
cp stage1_results/results-2/*  phase5_results/
cp stage2_results/*            phase5_results/
cp missing_arrays/*            phase5_results/
cp checkpoint_a/checkpoints/*  checkpoints/
cp checkpoint_b/*              checkpoints/

echo "phase5_results/ : $(ls phase5_results | wc -l) files   (expected 111)"
echo "checkpoints/    : $(ls checkpoints    | wc -l) files   (expected 13)"
echo
echo "Run scripts from the repository root, e.g.:"
echo "  python scripts/script_63_nll_mix_vs_selection.py"
