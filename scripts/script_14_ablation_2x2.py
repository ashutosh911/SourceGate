"""
script_14_ablation_2x2.py — 2×2 Factorial Ablation: Loss Type × Sampler Type

Isolates the independent contributions of SourceFormer's two training-procedure
innovations over MLP-HardCE:
  Factor A — Loss type:    hard CE  vs  soft KL divergence
  Factor B — Sampler type: uniform  vs  inverse-frequency weighted sampling

Four cells (architecture, optimizer, schedule, seeds identical across all):

  Cell 0  CE  + Uniform   → MLP-HardCE        (existing, macro ≈ 0.702)
  Cell 1  CE  + Weighted  → CE-Weighted        (NEW)
  Cell 2  KL  + Uniform   → SF Phase 3         (existing, macro ≈ 0.737)
  Cell 3  KL  + Weighted  → KL-Weighted        (NEW)

Cells 0 and 2 re-run here for a perfectly controlled comparison
(same RNG state, same early-stopping criterion, same evaluation).
Existing checkpoints are loaded if available (skip flag), so
re-running is fast — only cells 1 and 3 require new training.

Expected finding:
  Soft KL carries most of the +3.5 macro gain over MLP-HardCE.
  Weighted sampling provides an additional KG-specific boost (minority class).
  The interaction (KL + Weighted) shows whether the two factors compound.

USAGE:
  python script_14_ablation_2x2.py               # run all 4 cells
  python script_14_ablation_2x2.py --cells 1 3   # run only new cells
  python script_14_ablation_2x2.py --smoke        # 3 epochs, 1 seed

OUTPUT:
  checkpoints_ablation_2x2/  — per-cell per-seed checkpoints
  phase7_results/ablation_2x2_results.json  — full metrics
  phase7_results/ablation_2x2_table.txt     — printable comparison table

RUNTIME:
  Each cell × seed: ~2-4 min on RX 9060 XT (no LLM needed).
  All 4 cells × 3 seeds = ~24-48 min total.
"""

import json, time, gc, math, random, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3,
    SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS,
    EMBED_DIM,
)

# =============================================================================
# Config — identical to script_5b_k3_FINAL.py; do NOT change without justification
# =============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN      = 512
ENCODE_BATCH = 64
EPOCHS       = 50
LR           = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.2
BATCH_SIZE   = 128
PATIENCE     = 15
MIN_DELTA    = 0.005
WARMUP_FRAC  = 0.1
GRAD_CLIP    = 1.0
SEEDS        = [42, 123, 2026]

TRAIN_FILE = "mmrag_train.json"
DEV_FILE   = "mmrag_dev.json"
TEST_FILE  = "mmrag_test.json"
EMB_CACHE  = Path("query_emb_cache");         EMB_CACHE.mkdir(exist_ok=True)
CKPT_DIR   = Path("checkpoints_ablation_2x2"); CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("phase7_results");          RESULTS_DIR.mkdir(exist_ok=True)

# Existing checkpoint dirs for cells 0 and 2 (load instead of re-train)
EXISTING_CKPT = {
    "CE_Uniform":  Path("checkpoints"),           # MLP-HardCE — script_11
    "KL_Uniform":  Path("checkpoints"),           # SF Phase 3  — script_5b
}
EXISTING_PREFIX = {
    "CE_Uniform":  None,          # script_11 saves in-memory; no ckpt files
    "KL_Uniform":  "sourceformer_k3_seed",
}

# Cell definitions (loss_type, sampler_type, label)
CELLS = [
    ("CE", "Uniform",  "CE_Uniform"),    # cell 0: MLP-HardCE    (existing)
    ("CE", "Weighted", "CE_Weighted"),   # cell 1: CE + Weighted  (NEW)
    ("KL", "Uniform",  "KL_Uniform"),    # cell 2: SF Phase 3     (existing)
    ("KL", "Weighted", "KL_Weighted"),   # cell 3: KL + Weighted  (NEW)
]


# =============================================================================
# Utilities
# =============================================================================
def to_json_safe(obj):
    if isinstance(obj, dict):   return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


# =============================================================================
# Data loading — identical label construction to script_5b_k3_FINAL.py
# =============================================================================
def aggregated_type_scores(item):
    ts = {t: 0.0 for t in SOURCE_TYPES}
    for src, s in item["dataset_score"].items():
        if src in DATASET_TO_TYPE:
            ts[DATASET_TO_TYPE[src]] += float(s)
    return ts


def soft_target(item):
    ts = aggregated_type_scores(item)
    total = sum(ts.values())
    if total == 0:
        return None
    return np.array([ts[t] / total for t in SOURCE_TYPES], dtype=np.float32)


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    records, skipped = [], 0
    for item in data:
        soft = soft_target(item)
        if soft is None:
            skipped += 1
            continue
        records.append({
            "query":      item["query"],
            "soft_label": soft,
            "hard_label": int(np.argmax(soft)),
        })
    counts = Counter(r["hard_label"] for r in records)
    print(f"  {path}: {len(records):,} labelled ({skipped} skipped)")
    for idx, t in enumerate(SOURCE_TYPES):
        n = counts.get(idx, 0)
        print(f"    {t:6s}: {n:>5,}  ({100*n/len(records):.1f}%)")
    return records


# =============================================================================
# BGE encoding — reuses caches from script_5b
# =============================================================================
def encode_to_disk(records, cache_path):
    if cache_path.exists():
        return np.load(cache_path)
    from transformers import AutoTokenizer, AutoModel
    print(f"  Encoding {len(records)} queries → {cache_path} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bge = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    queries = [QUERY_PREFIX + r["query"] for r in records]
    n   = len(queries)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, n, ENCODE_BATCH), desc="BGE"):
            e   = min(s + ENCODE_BATCH, n)
            enc = tok(queries[s:e], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            emb = bge(**enc).last_hidden_state[:, 0]
            out[s:e] = F.normalize(emb.float(), p=2, dim=1).cpu().numpy()
    del bge, tok; gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    np.save(cache_path, out)
    return out


# =============================================================================
# Dataset
# =============================================================================
class RoutingDataset(Dataset):
    """Unified dataset returning (emb, soft, hard) for all four cells."""
    def __init__(self, records, embs):
        self.embs = torch.from_numpy(embs).float()
        self.soft = torch.tensor(
            np.stack([r["soft_label"] for r in records]), dtype=torch.float32
        )
        self.hard = torch.tensor(
            [r["hard_label"] for r in records], dtype=torch.long
        )

    def __len__(self):  return len(self.hard)

    def __getitem__(self, idx):
        return self.embs[idx], self.soft[idx], self.hard[idx]


def make_loader(dataset, sampler_type: str, hard_labels: np.ndarray,
                shuffle_val: bool = False) -> DataLoader:
    """
    Build a DataLoader for the given sampler type.
    sampler_type: 'Uniform' or 'Weighted'
    For val/test always returns a plain sequential loader.
    """
    if shuffle_val:
        # Validation loader — always sequential, no sampling
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

    if sampler_type == "Uniform":
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

    # Weighted: inverse-frequency weighting per sample
    counts     = Counter(hard_labels.tolist())
    n_total    = len(hard_labels)
    class_w    = {c: n_total / cnt for c, cnt in counts.items()}
    sample_w   = [class_w[int(hard_labels[i])] for i in range(n_total)]
    sampler    = WeightedRandomSampler(
        weights     = sample_w,
        num_samples = len(sample_w),
        replacement = True,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                      num_workers=0, pin_memory=(DEVICE == "cuda"))


# =============================================================================
# Loss functions
# =============================================================================
def kl_soft_loss(logits, soft_targets):
    """Soft KL against normalized dataset_score targets (script_5b §4.1)."""
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1).mean()


def ce_hard_loss(logits, soft_targets, hard_labels):
    """Standard cross-entropy on hard argmax labels (script_11 §3.2)."""
    return F.cross_entropy(logits, hard_labels)


def compute_loss(logits, soft, hard, loss_type: str):
    if loss_type == "KL":
        return kl_soft_loss(logits, soft)
    return ce_hard_loss(logits, soft, hard)


# =============================================================================
# LR schedule — identical to script_5b
# =============================================================================
def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Evaluation — identical to script_5b
# =============================================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_hard, all_soft = [], [], []
    for embs, soft, hard in loader:
        all_logits.append(model(embs.to(DEVICE)))
        all_hard.append(hard.to(DEVICE))
        all_soft.append(soft.to(DEVICE))
    logits = torch.cat(all_logits)
    hard   = torch.cat(all_hard)
    soft   = torch.cat(all_soft)
    loss   = kl_soft_loss(logits, soft).item()   # always KL for comparability
    preds  = logits.argmax(-1)
    acc    = (preds == hard).float().mean().item()
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (hard == idx)
        per_type[t] = (preds[mask] == idx).float().mean().item() if mask.sum() > 0 else float("nan")
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return loss, acc, per_type, macro


# =============================================================================
# Training loop — single seed, one cell
# =============================================================================
def train_one_seed(seed: int, loss_type: str, sampler_type: str,
                   cell_label: str,
                   train_ds, val_ds, train_hard_labels: np.ndarray,
                   smoke: bool = False) -> dict:

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    epochs_this = 3 if smoke else EPOCHS

    model     = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = make_loader(train_ds, sampler_type, train_hard_labels)
    val_loader   = make_loader(val_ds,   "Uniform",    None, shuffle_val=True)

    total_steps  = epochs_this * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler    = cosine_warmup(optimizer, warmup_steps, total_steps)

    best_macro     = 0.0
    best_acc       = 0.0
    best_per_type  = None
    best_epoch     = 0
    patience_ctr   = 0
    ckpt_path      = CKPT_DIR / f"{cell_label}_seed{seed}_best.pt"

    for epoch in range(1, epochs_this + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for embs, soft, hard in train_loader:
            embs, soft, hard = embs.to(DEVICE), soft.to(DEVICE), hard.to(DEVICE)
            optimizer.zero_grad()
            logits = model(embs)
            loss   = compute_loss(logits, soft, hard, loss_type)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            tr_loss    += loss.item() * hard.size(0)
            tr_correct += (logits.argmax(-1) == hard).sum().item()
            tr_n       += hard.size(0)

        _, vl_acc, vl_per_type, vl_macro = evaluate(model, val_loader)
        flag = ""
        if vl_macro > best_macro + MIN_DELTA:
            best_macro    = vl_macro
            best_acc      = vl_acc
            best_per_type = vl_per_type.copy()
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save({
                "epoch": epoch, "seed": seed,
                "loss_type": loss_type, "sampler_type": sampler_type,
                "cell_label": cell_label,
                "state_dict": model.state_dict(),
                "val_macro": vl_macro, "val_acc": vl_acc,
                "val_per_type": vl_per_type,
            }, ckpt_path)
            flag = " ← best"
        else:
            patience_ctr += 1

        if epoch <= 3 or epoch % 10 == 0 or flag or patience_ctr >= PATIENCE:
            type_str = "  ".join(
                f"{t}={vl_per_type.get(t, float('nan')):.3f}" for t in SOURCE_TYPES
            )
            print(f"    Ep {epoch:3d}  loss={tr_loss/tr_n:.4f}  "
                  f"vl_acc={vl_acc:.3f}  macro={vl_macro:.3f}  [{type_str}]{flag}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}.")
            break

    return {
        "seed":       seed,
        "best_epoch": best_epoch,
        "val_macro":  best_macro,
        "val_acc":    best_acc,
        "val_per_type": best_per_type,
    }


# =============================================================================
# Evaluate a trained checkpoint on the test set
# =============================================================================
@torch.no_grad()
def eval_on_test(ckpt_path: Path, test_ds) -> dict:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=(DEVICE == "cuda"))
    _, acc, per_type, macro = evaluate(model, test_loader)
    del model
    return {"test_acc": acc, "test_macro": macro, "test_per_type": per_type}


# =============================================================================
# Run one cell across all seeds
# =============================================================================
def run_cell(loss_type: str, sampler_type: str, cell_label: str,
             train_ds, val_ds, test_ds,
             train_hard_labels: np.ndarray,
             smoke: bool = False) -> dict:

    print(f"\n{'='*65}")
    print(f"  Cell: {cell_label}  |  loss={loss_type}  sampler={sampler_type}")
    print(f"{'='*65}")

    seeds_this = [SEEDS[0]] if smoke else SEEDS
    per_seed   = []

    for seed in seeds_this:
        print(f"\n  Seed {seed}:")
        ckpt_path = CKPT_DIR / f"{cell_label}_seed{seed}_best.pt"
        if ckpt_path.exists():
            print(f"    Checkpoint found — loading {ckpt_path}")
            # Only re-evaluate, skip training
            train_result = torch.load(ckpt_path, map_location="cpu")
            seed_result  = {
                "seed":        seed,
                "best_epoch":  train_result.get("epoch", -1),
                "val_macro":   train_result.get("val_macro",   float("nan")),
                "val_acc":     train_result.get("val_acc",     float("nan")),
                "val_per_type": train_result.get("val_per_type", {}),
            }
        else:
            seed_result = train_one_seed(
                seed, loss_type, sampler_type, cell_label,
                train_ds, val_ds, train_hard_labels, smoke=smoke,
            )

        # Test set evaluation
        test_result = eval_on_test(ckpt_path, test_ds)
        seed_result.update(test_result)
        per_seed.append(seed_result)

        print(f"    Val macro={seed_result['val_macro']:.4f}  "
              f"Test macro={seed_result['test_macro']:.4f}")

    # Aggregate across seeds
    test_macros   = [r["test_macro"] for r in per_seed]
    test_accs     = [r["test_acc"]   for r in per_seed]
    mean_macro    = float(np.mean(test_macros))
    std_macro     = float(np.std(test_macros))
    mean_acc      = float(np.mean(test_accs))

    per_type_agg = {}
    for t in SOURCE_TYPES:
        vals = [r["test_per_type"].get(t, float("nan")) for r in per_seed]
        valid = [v for v in vals if not np.isnan(v)]
        per_type_agg[t] = {
            "mean": float(np.mean(valid)) if valid else float("nan"),
            "std":  float(np.std(valid))  if valid else float("nan"),
        }

    print(f"\n  {cell_label}: test macro={mean_macro:.4f} ± {std_macro:.4f}  "
          f"acc={mean_acc:.4f}")
    type_str = "  ".join(
        f"{t}={per_type_agg[t]['mean']:.3f}±{per_type_agg[t]['std']:.3f}"
        for t in SOURCE_TYPES
    )
    print(f"  Per-type: [{type_str}]")

    return {
        "cell_label":    cell_label,
        "loss_type":     loss_type,
        "sampler_type":  sampler_type,
        "n_seeds":       len(per_seed),
        "test_macro":    mean_macro,
        "test_macro_std": std_macro,
        "test_acc":      mean_acc,
        "test_per_type": per_type_agg,
        "per_seed":      per_seed,
    }


# =============================================================================
# Print comparison table
# =============================================================================
def print_table(results: list) -> str:
    header = (
        f"\n{'='*80}\n"
        f"2×2 ABLATION: LOSS TYPE × SAMPLER TYPE  (K=3 test set)\n"
        f"{'='*80}\n"
        f"  Factor A — Loss:    Hard CE  vs  Soft KL divergence\n"
        f"  Factor B — Sampler: Uniform  vs  Inverse-freq weighted\n"
        f"{'='*80}\n"
        f"{'Cell':<20} {'Loss':>8} {'Sampler':>10} "
        f"{'Macro':>10} {'Text':>8} {'Table':>8} {'KG':>8}\n"
        f"{'-'*80}"
    )
    rows = [header]
    for r in results:
        pt  = r["test_per_type"]
        row = (
            f"{r['cell_label']:<20} {r['loss_type']:>8} {r['sampler_type']:>10} "
            f"{r['test_macro']:>9.4f}±{r['test_macro_std']:.3f} "
            f"{pt['text']['mean']:>7.3f} "
            f"{pt['table']['mean']:>7.3f} "
            f"{pt['kg']['mean']:>7.3f}"
        )
        rows.append(row)

    rows.append(f"{'='*80}")
    rows.append("\nFactor effects (approximate, on macro):")
    result_map = {r["cell_label"]: r for r in results}

    if all(k in result_map for k in ["CE_Uniform","CE_Weighted","KL_Uniform","KL_Weighted"]):
        ce_u  = result_map["CE_Uniform"]["test_macro"]
        ce_w  = result_map["CE_Weighted"]["test_macro"]
        kl_u  = result_map["KL_Uniform"]["test_macro"]
        kl_w  = result_map["KL_Weighted"]["test_macro"]
        # Main effect A (KL vs CE): average across sampler conditions
        effect_A = ((kl_u - ce_u) + (kl_w - ce_w)) / 2
        # Main effect B (Weighted vs Uniform): average across loss conditions
        effect_B = ((ce_w - ce_u) + (kl_w - kl_u)) / 2
        # Interaction
        interaction = (kl_w - kl_u) - (ce_w - ce_u)
        rows.append(f"  Main effect A (Soft KL over Hard CE):          {effect_A:+.4f}")
        rows.append(f"  Main effect B (Weighted over Uniform sampling): {effect_B:+.4f}")
        rows.append(f"  Interaction (A×B):                             {interaction:+.4f}")
        rows.append(f"\n  Interpretation:")
        if abs(effect_A) > abs(effect_B):
            rows.append(f"  → Soft KL is the dominant factor.")
        else:
            rows.append(f"  → Weighted sampling is the dominant factor.")
        if abs(interaction) > 0.005:
            rows.append(f"  → Interaction is non-trivial: factors are not independent.")
        else:
            rows.append(f"  → Interaction is small: factors compound approximately additively.")

    table_str = "\n".join(rows)
    print(table_str)
    return table_str


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Which cells to run (0=CE_Uniform, 1=CE_Weighted, 2=KL_Uniform, 3=KL_Weighted)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 3 epochs, 1 seed per cell."
    )
    args = parser.parse_args()

    cells_to_run = [CELLS[i] for i in args.cells]

    print("=" * 65)
    print("2×2 Factorial Ablation: Loss Type × Sampler Type")
    print("=" * 65)
    print(f"Cells to run: {[c[2] for c in cells_to_run]}")
    print(f"Seeds: {SEEDS if not args.smoke else [SEEDS[0]]}")
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    train_records = load_records(TRAIN_FILE)
    val_records   = load_records(DEV_FILE)
    test_records  = load_records(TEST_FILE)

    train_hard = np.array([r["hard_label"] for r in train_records])

    # Embeddings (reuse script_5b caches)
    print("\nEmbeddings (reusing cache if available):")
    train_embs = encode_to_disk(train_records, EMB_CACHE / "train_embs.npy")
    val_embs   = encode_to_disk(val_records,   EMB_CACHE / "val_embs.npy")
    test_embs  = encode_to_disk(test_records,  EMB_CACHE / "test_embs.npy")
    print(f"  train={train_embs.shape}  val={val_embs.shape}  test={test_embs.shape}")

    # Datasets
    train_ds = RoutingDataset(train_records, train_embs)
    val_ds   = RoutingDataset(val_records,   val_embs)
    test_ds  = RoutingDataset(test_records,  test_embs)

    # Class distribution (informative)
    counts = Counter(train_hard.tolist())
    print("\nTraining class distribution (affects weighted sampler):")
    for idx, t in enumerate(SOURCE_TYPES):
        n   = counts.get(idx, 0)
        w   = len(train_hard) / n if n > 0 else 0
        print(f"  {t:6s}: {n:>5,}  ({100*n/len(train_hard):.1f}%)  "
              f"→ inverse-freq weight = {w:.2f}x")

    # Run cells
    all_results = []
    t_start = time.time()

    for loss_type, sampler_type, cell_label in cells_to_run:
        result = run_cell(
            loss_type, sampler_type, cell_label,
            train_ds, val_ds, test_ds, train_hard,
            smoke=args.smoke,
        )
        all_results.append(result)

    # Print and save comparison table
    table_str = print_table(all_results)
    table_path = RESULTS_DIR / "ablation_2x2_table.txt"
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"\nTable saved to {table_path}")

    # Save full JSON results
    json_path = RESULTS_DIR / "ablation_2x2_results.json"
    with open(json_path, "w") as f:
        json.dump(to_json_safe({
            "description": "2x2 factorial ablation: loss_type x sampler_type",
            "factors": {
                "A": {"name": "loss_type",    "levels": ["CE", "KL"]},
                "B": {"name": "sampler_type", "levels": ["Uniform", "Weighted"]},
            },
            "cells": all_results,
            "total_time_min": (time.time() - t_start) / 60,
        }), f, indent=2)
    print(f"Full results saved to {json_path}")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
