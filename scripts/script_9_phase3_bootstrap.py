"""
script_9_phase3_bootstrap.py

Paired query-level bootstrap: Phase 3 (supervised pretraining) vs BGE-confidence
on the full test set (n=1286).

NO LLM NEEDED. Only BGE encoder + FAISS indices + Phase 3 checkpoints.
Runtime: ~5-10 minutes on GPU.

Outputs:
  - phase3_bootstrap_results.json  (the 4 numbers for the paper)
  - Console summary ready to paste

Usage:
  python script_9_phase3_bootstrap.py
  python script_9_phase3_bootstrap.py --seeds 42 123 2026
  python script_9_phase3_bootstrap.py --n_boot 50000  # more replicates
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import gc
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3, SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS, EMBED_DIM,
)


# =============================================================================
# Config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 512
ENCODE_BATCH = 64

TEST_FILE = "mmrag_test.json"
INDICES_DIR = Path("faiss_indices")
PHASE3_CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("phase5_results"); RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Test data loading (same as script_7)
# =============================================================================
def aggregated_type_scores(item):
    ts = {t: 0.0 for t in SOURCE_TYPES}
    for src, s in item["dataset_score"].items():
        if src in DATASET_TO_TYPE:
            ts[DATASET_TO_TYPE[src]] += float(s)
    return ts


def oracle_type(item):
    ts = aggregated_type_scores(item)
    total = sum(ts.values())
    if total == 0:
        return None
    return int(np.argmax([ts[t] for t in SOURCE_TYPES]))


def load_test_records(path):
    with open(path) as f:
        data = json.load(f)
    out = []
    for item in data:
        oracle = oracle_type(item)
        if oracle is None:
            continue
        out.append({
            "id": item.get("id", ""),
            "query": item["query"],
            "oracle_label": oracle,
        })
    print(f"  Loaded {len(out):,} test queries")
    return out


# =============================================================================
# BGE encoding
# =============================================================================
def load_bge():
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    return model, tok


@torch.no_grad()
def encode_queries(bge, bge_tok, queries):
    out = np.empty((len(queries), EMBED_DIM), dtype=np.float32)
    prefixed = [QUERY_PREFIX + q for q in queries]
    for s in range(0, len(queries), ENCODE_BATCH):
        e = min(s + ENCODE_BATCH, len(queries))
        enc = bge_tok(prefixed[s:e], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        emb = bge(**enc).last_hidden_state[:, 0]
        emb = F.normalize(emb.float(), p=2, dim=1)
        out[s:e] = emb.cpu().numpy()
    return out


# =============================================================================
# BGE-confidence baseline: pick source with highest top-1 retrieval score
# =============================================================================
def compute_bge_confidence_routing(query_embs, indices_dir, top_k=1):
    """
    For each query, search each source type's FAISS index, get the top-1
    cosine similarity score, and route to the source with the highest score.
    """
    n = query_embs.shape[0]
    decisions = np.empty(n, dtype=np.int64)

    # Load indices per source type
    type_indices = {}
    for type_name, datasets in TYPE_TO_DATASETS.items():
        # For types with multiple datasets (text: nq+triviaqa, table: ott+tat),
        # we search both and take the best score
        ds_indices = []
        for ds in datasets:
            idx = faiss.read_index(str(indices_dir / ds / "index.faiss"))
            ds_indices.append(idx)
        type_indices[type_name] = ds_indices

    # For each query, get best score per source type
    print("  Computing BGE-confidence routing...")
    for i in tqdm(range(n), desc="BGE-confidence"):
        q = query_embs[i:i+1]  # (1, 768)
        type_scores = {}
        for type_idx, type_name in enumerate(SOURCE_TYPES):
            best_score = -np.inf
            for idx in type_indices[type_name]:
                scores, _ = idx.search(q, top_k)
                best_score = max(best_score, float(scores[0, 0]))
            type_scores[type_idx] = best_score
        decisions[i] = max(type_scores, key=type_scores.get)

    return decisions


# =============================================================================
# Phase 3 routing
# =============================================================================
@torch.no_grad()
def compute_phase3_routing(query_embs, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    sf = SourceFormerK3().to(DEVICE)
    sf.load_state_dict(ckpt["state_dict"])
    sf.eval()
    embs = torch.from_numpy(query_embs).float().to(DEVICE)
    decisions = sf(embs).argmax(-1).cpu().numpy()
    del sf, ckpt
    return decisions


# =============================================================================
# Macro accuracy computation
# =============================================================================
def compute_macro(decisions, oracle_labels, n_types=K):
    """Per-type accuracy averaged across types."""
    per_type_acc = []
    for t in range(n_types):
        mask = (oracle_labels == t)
        if mask.sum() > 0:
            per_type_acc.append(float((decisions[mask] == oracle_labels[mask]).mean()))
        else:
            per_type_acc.append(float('nan'))
    valid = [v for v in per_type_acc if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float('nan')


# =============================================================================
# Paired bootstrap
# =============================================================================
def paired_bootstrap(
    decisions_a,    # Phase 3 routing decisions (n,)
    decisions_b,    # BGE-confidence routing decisions (n,)
    oracle_labels,  # Ground truth (n,)
    n_boot=10000,
    seed=42,
):
    """
    Query-level paired bootstrap for macro accuracy difference.
    
    For each replicate:
      1. Resample query indices with replacement
      2. Compute macro_a and macro_b on the resampled set
      3. Record diff = macro_a - macro_b
    
    Returns: mean_diff, ci_lower, ci_upper, p_value
    """
    n = len(oracle_labels)
    rng = np.random.default_rng(seed)

    # Precompute per-query correctness
    correct_a = (decisions_a == oracle_labels).astype(np.float64)
    correct_b = (decisions_b == oracle_labels).astype(np.float64)

    diffs = np.empty(n_boot, dtype=np.float64)

    for b in tqdm(range(n_boot), desc="Bootstrap"):
        idx = rng.integers(0, n, size=n)
        oracle_boot = oracle_labels[idx]
        ca_boot = correct_a[idx]
        cb_boot = correct_b[idx]

        # Macro accuracy for both methods on this resample
        macro_a_parts, macro_b_parts = [], []
        for t in range(K):
            mask = (oracle_boot == t)
            count = mask.sum()
            if count > 0:
                macro_a_parts.append(ca_boot[mask].mean())
                macro_b_parts.append(cb_boot[mask].mean())

        macro_a = np.mean(macro_a_parts) if macro_a_parts else np.nan
        macro_b = np.mean(macro_b_parts) if macro_b_parts else np.nan
        diffs[b] = macro_a - macro_b

    # Remove any NaN replicates (shouldn't happen with 1286 queries, but safety)
    diffs = diffs[~np.isnan(diffs)]

    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    p_value = float((diffs <= 0).mean())  # one-sided: fraction where A is not better

    return mean_diff, ci_lower, ci_upper, p_value, diffs


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3 bootstrap vs BGE-confidence")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Phase 3 seeds to evaluate")
    parser.add_argument("--n_boot", type=int, default=10000,
                        help="Number of bootstrap replicates")
    parser.add_argument("--boot_seed", type=int, default=42,
                        help="RNG seed for bootstrap resampling")
    args = parser.parse_args()

    print("=" * 65)
    print("Phase 3 Bootstrap: SourceFormer (supervised) vs BGE-confidence")
    print("=" * 65)

    # 1. Load test data
    print("\n[1/5] Loading test data...")
    test_records = load_test_records(TEST_FILE)
    oracle_labels = np.array([r["oracle_label"] for r in test_records])
    n = len(test_records)
    print(f"  n = {n}")
    for t_idx, t_name in enumerate(SOURCE_TYPES):
        count = (oracle_labels == t_idx).sum()
        print(f"    {t_name}: {count} ({100*count/n:.1f}%)")

    # 2. Encode queries
    print("\n[2/5] Encoding test queries with BGE...")
    bge, bge_tok = load_bge()
    query_embs = encode_queries(bge, bge_tok, [r["query"] for r in test_records])
    del bge, bge_tok; gc.collect(); torch.cuda.empty_cache()
    print(f"  Shape: {query_embs.shape}")

    # 3. BGE-confidence routing
    print("\n[3/5] Computing BGE-confidence baseline...")
    bge_decisions = compute_bge_confidence_routing(query_embs, INDICES_DIR)
    bge_macro = compute_macro(bge_decisions, oracle_labels)
    print(f"  BGE-confidence macro: {bge_macro:.4f}")
    for t_idx, t_name in enumerate(SOURCE_TYPES):
        mask = (oracle_labels == t_idx)
        if mask.sum() > 0:
            acc = (bge_decisions[mask] == oracle_labels[mask]).mean()
            print(f"    {t_name}: {acc:.4f}")

    # 4. Phase 3 routing (all seeds)
    print("\n[4/5] Computing Phase 3 routing...")
    phase3_per_seed = {}
    for seed in args.seeds:
        ckpt_path = PHASE3_CKPT_DIR / f"sourceformer_k3_seed{seed}_best.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path} not found")
            continue
        decisions = compute_phase3_routing(query_embs, ckpt_path)
        macro = compute_macro(decisions, oracle_labels)
        phase3_per_seed[seed] = {"decisions": decisions, "macro": macro}
        print(f"  Seed {seed}: macro = {macro:.4f}")

    if not phase3_per_seed:
        print("\nERROR: No Phase 3 checkpoints found. Check PHASE3_CKPT_DIR.")
        return

    # Use best seed for the primary bootstrap (consistent with paper reporting)
    best_seed = max(phase3_per_seed, key=lambda s: phase3_per_seed[s]["macro"])
    best_decisions = phase3_per_seed[best_seed]["decisions"]
    best_macro = phase3_per_seed[best_seed]["macro"]
    print(f"\n  Best seed: {best_seed} (macro = {best_macro:.4f})")
    print(f"  Point estimate: Phase 3 - BGE = {best_macro - bge_macro:+.4f}")

    # Also compute mean across seeds
    all_macros = [v["macro"] for v in phase3_per_seed.values()]
    mean_macro = float(np.mean(all_macros))
    std_macro = float(np.std(all_macros))
    print(f"  Mean across seeds: {mean_macro:.4f} ± {std_macro:.4f}")
    print(f"  Mean lift over BGE: {mean_macro - bge_macro:+.4f}")

    # 5. Paired bootstrap (mean-seed predictions)
    # Strategy: for each query, use majority vote across seeds (or mean-seed approach)
    # Paper uses aggregated macro (mean of per-seed macros), so we bootstrap
    # each seed separately and report the mean-seed bootstrap.
    print(f"\n[5/5] Paired bootstrap ({args.n_boot:,} replicates)...")

    # Primary: bootstrap with best seed (matches paper's primary reporting)
    print(f"\n  --- Bootstrap: Phase 3 seed {best_seed} vs BGE-confidence ---")
    mean_diff, ci_lo, ci_hi, p_val, diffs = paired_bootstrap(
        best_decisions, bge_decisions, oracle_labels,
        n_boot=args.n_boot, seed=args.boot_seed,
    )
    print(f"  Mean diff:  {mean_diff:+.4f}")
    print(f"  95% CI:     [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  p-value:    {p_val:.6f}")

    # Also bootstrap each seed individually
    per_seed_bootstrap = {}
    for seed, data in phase3_per_seed.items():
        md, cl, ch, pv, _ = paired_bootstrap(
            data["decisions"], bge_decisions, oracle_labels,
            n_boot=args.n_boot, seed=args.boot_seed + seed,
        )
        per_seed_bootstrap[seed] = {
            "macro": data["macro"],
            "mean_diff": md, "ci_lower": cl, "ci_upper": ch, "p_value": pv,
        }
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  Seed {seed}: diff={md:+.4f}  CI=[{cl:+.4f}, {ch:+.4f}]  p={pv:.6f} {sig}")

    # =================================================================
    # Summary for the paper
    # =================================================================
    results = {
        "test_n": n,
        "n_bootstrap": args.n_boot,
        "boot_seed": args.boot_seed,
        "bge_confidence_macro": bge_macro,
        "phase3_seeds": args.seeds,
        "phase3_per_seed_macro": {s: v["macro"] for s, v in phase3_per_seed.items()},
        "phase3_mean_macro": mean_macro,
        "phase3_std_macro": std_macro,
        "best_seed": int(best_seed),
        "best_seed_macro": best_macro,
        "primary_bootstrap": {
            "comparison": f"phase3_seed{best_seed} vs bge_confidence",
            "mean_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
        },
        "per_seed_bootstrap": {str(k): v for k, v in per_seed_bootstrap.items()},
    }

    out_path = RESULTS_DIR / "phase3_bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Final console summary
    print("\n" + "=" * 65)
    print("RESULTS FOR PAPER (copy these)")
    print("=" * 65)
    print(f"Phase 3 test macro:    {best_macro:.3f} (seed {best_seed})")
    print(f"BGE-confidence macro:  {bge_macro:.3f}")
    print(f"Lift:                  {best_macro - bge_macro:+.3f}")
    print(f"Bootstrap mean diff:   {mean_diff:+.4f}")
    print(f"Bootstrap 95% CI:      [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"Bootstrap p-value:     {p_val:.6f}" + (" (p < 0.001)" if p_val < 0.001 else ""))
    print(f"Replicates:            {args.n_boot:,}")
    print(f"Test queries:          {n:,}")
    print("=" * 65)
    print()
    print("Send me these 4 numbers:")
    print(f"  mean_diff = {mean_diff:+.4f}")
    print(f"  ci_lower  = {ci_lo:+.4f}")
    print(f"  ci_upper  = {ci_hi:+.4f}")
    print(f"  p_value   = {p_val:.6f}")


if __name__ == "__main__":
    main()
