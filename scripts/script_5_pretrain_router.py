"""
Script 5: Pre-train SourceFormer with α=1.0 (routing supervision only).

WHY PRE-TRAIN BEFORE JOINT TRAINING?
  Cold-starting joint training with both L_route and L_ans simultaneously
  causes routing collapse: SourceFormer learns to always select the most
  common source because L_ans gradients are too noisy at init to teach
  source discrimination.

  Pre-training with α=1.0 gives SourceFormer a warm start with clean
  oracle supervision. Phase 5 joint training then only needs to fine-tune
  an already-capable router, not bootstrap one from scratch.

WHAT THIS SCRIPT DOES:
  1. Extract oracle routing labels from mmrag_dev.json (and train split).
  2. Build a dataset of (query_embedding, source_label) pairs.
  3. Pre-train MLPSourceFormer + TransformerSourceFormer with cross-entropy.
  4. Verify Gumbel-Softmax gradient bridge (Ablation 5 — gradient norm check).
  5. Save checkpoints ready for Phase 5 joint training.

INPUTS:
  - mmrag_dev.json          (from your existing workflow)
  - mmrag_train.json        (or whichever split has routing labels)
  - query_embeddings/       (optional: cache from script_4's encode_queries)
  - faiss_indices/          (not used here — just routing pre-train)

OUTPUTS:
  - checkpoints/sourceformer_mlp_best.pt
  - checkpoints/sourceformer_transformer_best.pt
  - routing_pretrain_log.json           (loss/accuracy curves per epoch)
  - gradient_bridge_verification.json   (Ablation 5 results)
"""

import json
import time
import gc
import math
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sourceformer import (
    MLPSourceFormer, TransformerSourceFormer, build_sourceformer,
    gumbel_softmax, SOURCES, SOURCE_TO_IDX, K, EMBED_DIM,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN      = 512
ENCODE_BATCH = 64

# Training
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 128
PATIENCE     = 5          # early stopping
WARMUP_FRAC  = 0.1        # first 10% of steps: linear LR warmup
GRAD_CLIP    = 1.0
SEEDS        = [42, 123, 2026]

# Paths
DEV_FILE     = "mmrag_dev.json"
TRAIN_FILE   = "mmrag_train.json"    # may not exist yet; dev-only fallback below
CKPT_DIR     = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
EMB_CACHE    = Path("query_emb_cache")   # cache encoded queries to avoid re-encoding
EMB_CACHE.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Step 1 — Oracle routing label extraction
# ---------------------------------------------------------------------------

def oracle_label(item: dict) -> int | None:
    """
    Return the integer source index for the oracle source of this item,
    or None if no source has a positive score (unanswerable / skip).

    Ties broken by source priority order (matching your script_4 logic).
    """
    scores  = item["dataset_score"]
    max_s   = max(scores.values())
    if max_s == 0:
        return None

    # All sources tied at max — use first by SOURCES order
    winners = [src for src in SOURCES if scores.get(src, 0) == max_s]
    return SOURCE_TO_IDX[winners[0]]


def load_routing_dataset(path: str) -> list[dict]:
    """
    Load a mmRAG JSON file and return list of:
        {"query": str, "label": int, "dataset_score": dict}
    Skips items with no positive-score source.
    """
    with open(path) as f:
        data = json.load(f)

    records = []
    skipped = 0
    for item in data:
        label = oracle_label(item)
        if label is None:
            skipped += 1
            continue
        records.append({
            "query":         item["query"],
            "label":         label,
            "dataset_score": item["dataset_score"],
        })

    print(f"  Loaded {len(records):,} labelled queries "
          f"({skipped} skipped — no positive source).")

    # Class distribution
    counts = Counter(r["label"] for r in records)
    print("  Label distribution:")
    for idx, src in enumerate(SOURCES):
        n = counts.get(idx, 0)
        print(f"    {src:10s}: {n:>5,}  ({100*n/len(records):.1f}%)")

    return records


print(f"\n=== Step 1: Loading routing labels ===")
dev_records = load_routing_dataset(DEV_FILE)

# If train split exists, use it; otherwise split dev 80/20
if Path(TRAIN_FILE).exists():
    print(f"\nLoading train split: {TRAIN_FILE}")
    train_records = load_routing_dataset(TRAIN_FILE)
    val_records   = dev_records
else:
    print(f"\n{TRAIN_FILE} not found — splitting dev 80/20 for train/val.")
    random.seed(42)
    shuffled = dev_records.copy()
    random.shuffle(shuffled)
    split     = int(0.8 * len(shuffled))
    train_records = shuffled[:split]
    val_records   = shuffled[split:]
    print(f"  Train: {len(train_records):,}  Val: {len(val_records):,}")


# ---------------------------------------------------------------------------
# Step 2 — Encode queries with BGE (with disk cache)
# ---------------------------------------------------------------------------

def encode_queries_to_disk(records: list[dict], cache_path: Path,
                            tok, model) -> np.ndarray:
    """
    Encode queries and save to disk. On subsequent runs, loads from cache.
    Returns (N, 768) float32 array.
    """
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    queries  = [QUERY_PREFIX + r["query"] for r in records]
    n        = len(queries)
    out      = np.empty((n, EMBED_DIM), dtype=np.float32)

    print(f"  Encoding {n:,} queries on {DEVICE}...")
    with torch.inference_mode():
        for start in tqdm(range(0, n, ENCODE_BATCH), desc="encoding"):
            end  = min(start + ENCODE_BATCH, n)
            enc  = tok(queries[start:end], padding=True, truncation=True,
                       max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            emb  = model(**enc).last_hidden_state[:, 0]
            emb  = F.normalize(emb, p=2, dim=1)
            out[start:end] = emb.cpu().numpy()

    np.save(cache_path, out)
    print(f"  Saved to {cache_path}")
    return out


print(f"\n=== Step 2: Encoding queries ===")
print(f"Loading {MODEL_NAME}...")
tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
bge   = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE).eval()

train_embs = encode_queries_to_disk(
    train_records, EMB_CACHE / "train_embs.npy", tok, bge)
val_embs   = encode_queries_to_disk(
    val_records, EMB_CACHE / "val_embs.npy", tok, bge)

# Free BGE — don't need it for SourceFormer training
del bge, tok
gc.collect()
torch.cuda.empty_cache()
print("BGE encoder freed.")


# ---------------------------------------------------------------------------
# Step 3 — PyTorch Dataset
# ---------------------------------------------------------------------------

class RoutingDataset(Dataset):
    def __init__(self, records: list[dict], embs: np.ndarray):
        assert len(records) == len(embs)
        self.embs   = torch.from_numpy(embs).float()
        self.labels = torch.tensor([r["label"] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]


train_ds = RoutingDataset(train_records, train_embs)
val_ds   = RoutingDataset(val_records,   val_embs)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

print(f"\nTrain batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")


# ---------------------------------------------------------------------------
# Step 4 — Training utilities
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int,
                                    num_training_steps: int):
    """Linear warmup → cosine decay LR schedule."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """Returns (loss, accuracy, per-source accuracy dict)."""
    loss     = F.cross_entropy(logits, labels)
    preds    = logits.argmax(dim=-1)
    correct  = (preds == labels).float()
    acc      = correct.mean().item()

    per_src  = {}
    for idx, src in enumerate(SOURCES):
        mask = (labels == idx)
        if mask.sum() > 0:
            per_src[src] = correct[mask].mean().item()
        else:
            per_src[src] = float("nan")

    return loss, acc, per_src


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    all_logits, all_labels = [], []
    for embs, labels in loader:
        embs, labels = embs.to(DEVICE), labels.to(DEVICE)
        all_logits.append(model(embs))
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss, acc, per_src = compute_metrics(logits, labels)
    return loss.item(), acc, per_src


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler, epoch: int):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for embs, labels in loader:
        embs, labels = embs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(embs)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss    += loss.item() * labels.size(0)
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_n       += labels.size(0)

    return total_loss / total_n, total_correct / total_n


def train_sourceformer(variant: str, seed: int):
    """Full training run for one variant + seed. Returns best val accuracy."""
    print(f"\n{'='*60}")
    print(f"Training {variant.upper()} SourceFormer  |  seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model     = build_sourceformer(variant).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    best_val_acc  = 0.0
    best_ckpt     = CKPT_DIR / f"sourceformer_{variant}_seed{seed}_best.pt"
    patience_ctr  = 0
    log           = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        vl_loss, vl_acc, vl_per_src = evaluate(model, val_loader)
        elapsed = time.time() - t0

        log.append({
            "epoch":          epoch,
            "train_loss":     tr_loss,
            "train_acc":      tr_acc,
            "val_loss":       vl_loss,
            "val_acc":        vl_acc,
            "val_per_source": vl_per_src,
            "lr":             scheduler.get_last_lr()[0],
        })

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":     epoch,
                "variant":   variant,
                "seed":      seed,
                "state_dict": model.state_dict(),
                "val_acc":   vl_acc,
                "val_per_source": vl_per_src,
            }, best_ckpt)
            patience_ctr = 0
            flag = "  ← best"
        else:
            patience_ctr += 1

        # Per-source breakdown string
        src_str = "  ".join(
            f"{s}={vl_per_src.get(s, float('nan')):.3f}" for s in SOURCES)

        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.3f}  "
              f"[{src_str}]  {elapsed:.1f}s{flag}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\n  Best val accuracy: {best_val_acc:.4f}  → {best_ckpt}")
    return best_val_acc, log, best_ckpt


# ---------------------------------------------------------------------------
# Step 5 — Run pre-training for both variants across seeds
# ---------------------------------------------------------------------------

all_logs = {}

for variant in ("mlp", "transformer"):
    variant_logs  = []
    variant_accs  = []
    for seed in SEEDS:
        best_acc, log, ckpt = train_sourceformer(variant, seed)
        variant_accs.append(best_acc)
        variant_logs.append({"seed": seed, "best_val_acc": best_acc, "epochs": log})

    mean_acc = np.mean(variant_accs)
    std_acc  = np.std(variant_accs)
    print(f"\n  {variant.upper()} over seeds: "
          f"mean={mean_acc:.4f}  std={std_acc:.4f}  "
          f"runs={variant_accs}")
    all_logs[variant] = {
        "mean_val_acc": mean_acc,
        "std_val_acc":  std_acc,
        "runs":         variant_logs,
    }

# Save training logs
log_path = Path("routing_pretrain_log.json")
with open(log_path, "w") as f:
    json.dump(all_logs, f, indent=2)
print(f"\nTraining logs saved to {log_path}")


# ---------------------------------------------------------------------------
# Step 6 — Gradient bridge verification  (Ablation 5)
# ---------------------------------------------------------------------------
# Load the best MLP checkpoint and verify that with α=0 (answer-loss only),
# gradients still flow back through Gumbel-Softmax to SourceFormer params.
#
# This is your mechanistic proof that the differentiable bridge is functional.
# Report this number in Section 4 of your paper ("Ablation 5: Gradient flow").
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Ablation 5: Gradient Bridge Verification")
print(f"{'='*60}")
print("Setting up dummy answer loss to verify ∂L_ans/∂SourceFormer ≠ 0...")

# Reload best MLP model (seed 42)
torch.manual_seed(42)
sf_model = build_sourceformer("mlp").to(DEVICE)

# Find best checkpoint for seed 42
best_mlp_ckpt = CKPT_DIR / "sourceformer_mlp_seed42_best.pt"
if best_mlp_ckpt.exists():
    ckpt_data = torch.load(best_mlp_ckpt, map_location=DEVICE)
    sf_model.load_state_dict(ckpt_data["state_dict"])
    print(f"  Loaded from {best_mlp_ckpt}")
else:
    print("  Using randomly initialized model (no checkpoint found).")

sf_model.train()

# Use a batch of validation embeddings
embs_batch, labels_batch = next(iter(val_loader))
embs_batch  = embs_batch[:32].to(DEVICE)
labels_batch = labels_batch[:32].to(DEVICE)

tau = 1.0    # τ at start of joint training

# Forward through SourceFormer → Gumbel-Softmax
logits  = sf_model(embs_batch)              # (B, K)
routing = gumbel_softmax(logits, tau=tau, hard=True)   # (B, K)

# Simulate a differentiable "answer quality" signal.
# In Phase 5 this will be AnswerFormer's LM loss; here we use a proxy:
# a learnable score matrix that scores each source independently.
# L_ans = -mean(routing · source_quality_scores)
# This mimics the gradient path: L_ans → routing → logits → SourceFormer params.
torch.manual_seed(0)
source_quality = torch.randn(K, requires_grad=False).to(DEVICE)
L_ans_proxy = -(routing * source_quality.unsqueeze(0)).sum(dim=-1).mean()

# Backward
L_ans_proxy.backward()

# Measure gradient norms at each SourceFormer parameter
param_grad_norms = []
for name, param in sf_model.named_parameters():
    if param.grad is not None:
        gnorm = param.grad.norm().item()
        param_grad_norms.append((name, gnorm))

total_gnorm = sum(g for _, g in param_grad_norms)
nonzero     = sum(1 for _, g in param_grad_norms if g > 1e-10)

print(f"\n  Gradient norms from L_ans proxy (α=0 setting):")
for name, gnorm in param_grad_norms:
    bar = "█" * min(int(gnorm * 200), 40)
    print(f"    {name:45s}: {gnorm:.6f}  {bar}")

print(f"\n  Total gradient norm:          {total_gnorm:.6f}")
print(f"  Parameters with grad > 0:     {nonzero}/{len(param_grad_norms)}")

if total_gnorm > 1e-8:
    bridge_status = "PASS"
    print(f"\n  ✓ GRADIENT BRIDGE VERIFIED — ∂L_ans/∂SourceFormer ≠ 0")
    print(f"    Total norm: {total_gnorm:.6f}  (report this in Ablation 5)")
else:
    bridge_status = "FAIL"
    print(f"\n  ✗ GRADIENT BRIDGE FAILED — gradients are zero!")
    print("    Check Gumbel-Softmax implementation (hard=True + detach trick).")

# Save verification results
verification = {
    "bridge_status":      bridge_status,
    "total_grad_norm":    total_gnorm,
    "nonzero_params":     nonzero,
    "total_params":       len(param_grad_norms),
    "tau_used":           tau,
    "per_param_grad_norm": {name: gnorm for name, gnorm in param_grad_norms},
}
with open("gradient_bridge_verification.json", "w") as f:
    json.dump(verification, f, indent=2)
print(f"\n  Verification results saved to gradient_bridge_verification.json")


# ---------------------------------------------------------------------------
# Step 7 — Final summary + Phase 5 handoff
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("PHASE 3 SUMMARY")
print(f"{'='*60}")

print(f"\nModel performance (val routing accuracy):")
for variant, res in all_logs.items():
    print(f"  {variant:15s}: {res['mean_val_acc']:.4f} ± {res['std_val_acc']:.4f}")

# Determine which variant to carry into Phase 5
best_variant = max(all_logs, key=lambda v: all_logs[v]["mean_val_acc"])
best_overall = all_logs[best_variant]["mean_val_acc"]

print(f"\nRecommended variant for Phase 5: {best_variant.upper()} "
      f"(mean val acc = {best_overall:.4f})")

# Find best checkpoint across seeds for that variant
best_ckpt_overall = None
best_acc_overall  = 0.0
for seed in SEEDS:
    ckpt = CKPT_DIR / f"sourceformer_{best_variant}_seed{seed}_best.pt"
    if ckpt.exists():
        data = torch.load(ckpt, map_location="cpu")
        if data["val_acc"] > best_acc_overall:
            best_acc_overall  = data["val_acc"]
            best_ckpt_overall = ckpt

print(f"Best single checkpoint: {best_ckpt_overall}  (val_acc={best_acc_overall:.4f})")

# Phase 5 readiness check
ROUTING_THRESHOLD = 0.80
if best_overall >= ROUTING_THRESHOLD:
    print(f"\n✓ Routing accuracy {best_overall:.4f} ≥ {ROUTING_THRESHOLD} threshold.")
    print("  Ready to proceed to Phase 5 (joint training).")
    print(f"  Load this checkpoint at the start of Phase 5:")
    print(f"    checkpoint = torch.load('{best_ckpt_overall}')")
    print(f"    sf_model   = build_sourceformer('{best_variant}')")
    print(f"    sf_model.load_state_dict(checkpoint['state_dict'])")
else:
    print(f"\n✗ Routing accuracy {best_overall:.4f} < {ROUTING_THRESHOLD} threshold.")
    print("  Do NOT proceed to Phase 5 yet. Diagnostics:")
    print("  1. Check per-source val_acc in routing_pretrain_log.json")
    print("  2. Sources with low per-source accuracy need:")
    print("     - More training examples (check class imbalance in Step 1 output)")
    print("     - Weighted sampling: add class_weight to F.cross_entropy")
    print("     - Longer training (increase EPOCHS or reduce early stopping PATIENCE)")
    print("  3. If kg accuracy is worst, consider merging kg+triviaqa into one class")
    print("     and running a 3-way classifier first.")
