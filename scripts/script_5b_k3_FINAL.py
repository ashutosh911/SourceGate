"""
Script 5b (PATCHED v2): Pre-train SourceFormer with K=3 source types.

Changes from v1:
  4. Macro-per-type early stopping (instead of overall accuracy). Prevents
     the model from sacrificing KG accuracy to gain marginal text/table.
  5. Patience increased to 15 with min_delta=0.005. Prevents premature
     early stopping at epochs 3-6 that was killing KG learning.

  Overall accuracy is still reported as the headline metric, but the
  CHECKPOINT saved is the one with best macro per-type accuracy. This
  matters because the dev set has 51.8% label ambiguity, so overall
  accuracy is a noisy signal compared to balanced per-class performance.
"""

import json, time, gc, math, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3, gumbel_softmax,
    SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS,
    EMBED_DIM,
)


# ---------- utilities ----------
def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


print(f"Source taxonomy: K={K}")
for t in SOURCE_TYPES:
    print(f"  {t:5s} ← {', '.join(TYPE_TO_DATASETS[t])}")


# ---------- config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 512
ENCODE_BATCH = 64
EPOCHS = 50            # increased from 40 to give patience headroom
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.2
BATCH_SIZE = 128
PATIENCE = 15          # PATCH: was 8 — too aggressive, killed KG learning
MIN_DELTA = 0.005      # PATCH: require ≥0.5% improvement to count as "better"
WARMUP_FRAC = 0.1
GRAD_CLIP = 1.0
SEEDS = [42, 123, 2026]
DEV_FILE = "mmrag_dev.json"
TRAIN_FILE = "mmrag_train.json"
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
EMB_CACHE = Path("query_emb_cache"); EMB_CACHE.mkdir(exist_ok=True)

print(f"\nDevice: {DEVICE}")


# ---------- soft label construction ----------
def aggregated_type_scores(item):
    type_scores = {t: 0.0 for t in SOURCE_TYPES}
    for src, s in item["dataset_score"].items():
        if src in DATASET_TO_TYPE:
            type_scores[DATASET_TO_TYPE[src]] += float(s)
    return type_scores


def soft_target(item):
    ts = aggregated_type_scores(item)
    total = sum(ts.values())
    if total == 0:
        return None
    return np.array([ts[t] / total for t in SOURCE_TYPES], dtype=np.float32)


def hard_label_from_soft(soft):
    return int(np.argmax(soft))


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    records, skipped = [], 0
    for item in data:
        soft = soft_target(item)
        if soft is None:
            skipped += 1
            continue
        records.append({
            "query": item["query"],
            "soft_label": soft,
            "hard_label": hard_label_from_soft(soft),
        })
    counts = Counter(r["hard_label"] for r in records)
    print(f"  {len(records):,} labelled ({skipped} skipped)")
    for idx, t in enumerate(SOURCE_TYPES):
        n = counts.get(idx, 0)
        print(f"    {t:6s}: {n:>5,} ({100*n/len(records):.1f}%)")
    return records


print("\n=== Step 1: Loading K=3 routing labels (aggregated, soft) ===")
dev_records = load_data(DEV_FILE)
if Path(TRAIN_FILE).exists():
    print(f"\nUsing {TRAIN_FILE} as training split.")
    train_records = load_data(TRAIN_FILE)
    val_records = dev_records
else:
    print(f"\n{TRAIN_FILE} not found — 80/20 split of dev.")
    random.seed(42)
    shuffled = dev_records.copy()
    random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_records, val_records = shuffled[:split], shuffled[split:]
    print(f"  Train: {len(train_records):,} Val: {len(val_records):,}")


# ---------- baselines ----------
print("\n=== Step 1b: Baseline accuracies on validation set ===")
val_hard = np.array([r["hard_label"] for r in val_records])
counts_val = Counter(val_hard.tolist())

majority_class = int(max(counts_val, key=counts_val.get))
majority_acc = (val_hard == majority_class).mean()

rng = np.random.default_rng(42)
random_preds = rng.integers(0, K, size=len(val_hard))
random_acc = (random_preds == val_hard).mean()

# Macro baseline = mean per-class recall under majority prediction
# (will be 1/K for majority class and 0 for others, so macro = 1/K = 0.333)
majority_per_type = {SOURCE_TYPES[i]: (1.0 if i == majority_class else 0.0) for i in range(K)}
majority_macro = np.mean(list(majority_per_type.values()))

unambiguous = sum(1 for r in val_records if (r["soft_label"] > 0.99).any())
ambiguity_rate = 1.0 - unambiguous / len(val_records)

print(f"  Majority-class ({SOURCE_TYPES[majority_class]}):")
print(f"    overall acc: {majority_acc:.4f}")
print(f"    macro acc:   {majority_macro:.4f}")
print(f"  Uniform random overall acc:  {random_acc:.4f}")
print(f"  Label ambiguity rate (val):  {ambiguity_rate:.3f}")
print(f"  → SourceFormer must clearly beat both metrics.")


# ---------- query encoding (cached) ----------
def encode_to_disk(records, cache_path, tok, model):
    if cache_path.exists():
        print(f"  Cache hit: {cache_path}")
        return np.load(cache_path)
    queries = [QUERY_PREFIX + r["query"] for r in records]
    n = len(queries)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, n, ENCODE_BATCH), desc="encoding"):
            e = min(s + ENCODE_BATCH, n)
            enc = tok(queries[s:e], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            emb = model(**enc).last_hidden_state[:, 0]
            emb = F.normalize(emb.float(), p=2, dim=1)
            out[s:e] = emb.cpu().numpy()
    np.save(cache_path, out)
    return out


print(f"\n=== Step 2: Encoding queries ===")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bge = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
train_embs = encode_to_disk(train_records, EMB_CACHE / "train_embs.npy", tok, bge)
val_embs = encode_to_disk(val_records, EMB_CACHE / "val_embs.npy", tok, bge)
del bge, tok; gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()


# ---------- dataset ----------
class RoutingDataset(Dataset):
    def __init__(self, records, embs):
        self.embs = torch.from_numpy(embs).float()
        self.soft = torch.tensor(np.stack([r["soft_label"] for r in records]),
                                 dtype=torch.float32)
        self.hard = torch.tensor([r["hard_label"] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.hard)

    def __getitem__(self, idx):
        return self.embs[idx], self.soft[idx], self.hard[idx]


train_ds = RoutingDataset(train_records, train_embs)
val_ds = RoutingDataset(val_records, val_embs)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=(DEVICE == "cuda"))


# ---------- KL loss against soft targets ----------
def kl_soft_loss(logits, soft_targets, eps=1e-8):
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1).mean()


# ---------- LR schedule ----------
def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------- evaluation: now also returns macro per-type ----------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_hard, all_soft = [], [], []
    for embs, soft, hard in loader:
        all_logits.append(model(embs.to(DEVICE)))
        all_hard.append(hard.to(DEVICE))
        all_soft.append(soft.to(DEVICE))
    logits = torch.cat(all_logits)
    hard = torch.cat(all_hard)
    soft = torch.cat(all_soft)
    loss = kl_soft_loss(logits, soft).item()
    preds = logits.argmax(-1)
    acc = (preds == hard).float().mean().item()
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (hard == idx)
        per_type[t] = (preds[mask] == idx).float().mean().item() if mask.sum() > 0 else float("nan")
    # PATCH: macro = mean of per-type accuracies (ignoring NaN)
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return loss, acc, per_type, macro


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for embs, soft, hard in loader:
        embs = embs.to(DEVICE)
        soft = soft.to(DEVICE)
        hard = hard.to(DEVICE)
        optimizer.zero_grad()
        logits = model(embs)
        loss = kl_soft_loss(logits, soft)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * hard.size(0)
        total_correct += (logits.argmax(-1) == hard).sum().item()
        n += hard.size(0)
    return total_loss / n, total_correct / n


def train_k3(seed):
    print(f"\n{'=' * 55}\nK=3 SourceFormer | seed={seed}\n{'=' * 55}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = cosine_warmup(optimizer, warmup_steps, total_steps)

    # PATCH: track BOTH overall and macro; checkpoint on macro
    best_val_macro = 0.0
    best_val_acc_at_best_macro = 0.0
    best_per_type_at_best_macro = None
    best_epoch = 0
    best_ckpt = CKPT_DIR / f"sourceformer_k3_seed{seed}_best.pt"
    patience_ctr = 0
    log = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
        vl_loss, vl_acc, vl_per_type, vl_macro = evaluate(model, val_loader)
        elapsed = time.time() - t0
        log.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc, "val_macro": vl_macro,
            "val_per_type": vl_per_type, "lr": scheduler.get_last_lr()[0],
        })
        flag = ""
        # PATCH: early stopping based on MACRO with min_delta
        if vl_macro > best_val_macro + MIN_DELTA:
            best_val_macro = vl_macro
            best_val_acc_at_best_macro = vl_acc
            best_per_type_at_best_macro = vl_per_type
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "seed": seed, "k": K,
                "source_types": SOURCE_TYPES, "type_to_datasets": TYPE_TO_DATASETS,
                "state_dict": model.state_dict(),
                "val_acc": vl_acc, "val_macro": vl_macro,
                "val_per_type": vl_per_type,
                "loss_type": "kl_soft",
                "selection_criterion": "macro_per_type",
            }, best_ckpt)
            patience_ctr = 0
            flag = "  ← best (macro)"
        else:
            patience_ctr += 1
        type_str = "  ".join(f"{t}={vl_per_type.get(t, float('nan')):.3f}" for t in SOURCE_TYPES)
        print(f"  Ep {epoch:3d} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
              f"vl_acc={vl_acc:.3f} vl_macro={vl_macro:.3f} [{type_str}] {elapsed:.1f}s{flag}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE}).")
            break

    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Best val macro: {best_val_macro:.4f}")
    print(f"  Val acc at best macro: {best_val_acc_at_best_macro:.4f}")
    print(f"  Per-type: {best_per_type_at_best_macro}")
    return {
        "macro": best_val_macro,
        "acc": best_val_acc_at_best_macro,
        "per_type": best_per_type_at_best_macro,
        "best_epoch": best_epoch,
    }, log


# ---------- run all seeds ----------
all_logs = {}
all_results = []
for seed in SEEDS:
    result, log = train_k3(seed)
    all_results.append(result)
    all_logs[f"seed_{seed}"] = {**result, "epochs": log}

mean_macro = np.mean([r["macro"] for r in all_results])
std_macro = np.std([r["macro"] for r in all_results])
mean_acc = np.mean([r["acc"] for r in all_results])
std_acc = np.std([r["acc"] for r in all_results])

# Per-type means across seeds
per_type_mean = {}
for t in SOURCE_TYPES:
    vals = [r["per_type"][t] for r in all_results
            if r["per_type"] is not None and not np.isnan(r["per_type"].get(t, float("nan")))]
    per_type_mean[t] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

with open("routing_pretrain_k3_log.json", "w") as f:
    json.dump(to_json_safe(all_logs), f, indent=2)
print("\nLogs saved to routing_pretrain_k3_log.json")


# ---------- gradient bridge verification ----------
print(f"\n{'=' * 55}\nAblation 5: Gradient bridge (K=3)\n{'=' * 55}")
best_seed = SEEDS[int(np.argmax([r["macro"] for r in all_results]))]
ckpt_data = torch.load(CKPT_DIR / f"sourceformer_k3_seed{best_seed}_best.pt", map_location=DEVICE)
sf = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
sf.load_state_dict(ckpt_data["state_dict"])
sf.train()
embs_b, _, _ = next(iter(val_loader))
embs_b = embs_b[:32].to(DEVICE)
logits = sf(embs_b)
routing = gumbel_softmax(logits, tau=1.0, hard=True)
source_quality = torch.randn(K).to(DEVICE)
L_ans_proxy = -(routing * source_quality).sum(-1).mean()
L_ans_proxy.backward()
total_gnorm = sum(p.grad.norm().item() for p in sf.parameters() if p.grad is not None)
nonzero = sum(1 for p in sf.parameters() if p.grad is not None and p.grad.norm().item() > 1e-10)
print(f"  Total grad norm from L_ans: {total_gnorm:.6f}")
print(f"  Params with grad > 0: {nonzero}/{sum(1 for _ in sf.parameters())}")
bridge_ok = total_gnorm > 1e-8
print("  ✓ GRADIENT BRIDGE VERIFIED" if bridge_ok else "  ✗ GRADIENT BRIDGE FAILED")


# ---------- summary ----------
print(f"\n{'=' * 55}\nK=3 PRE-TRAINING SUMMARY (macro selection)\n{'=' * 55}")
print(f"\nSelection criterion: macro per-type accuracy")
print("\nMacro per-type acc across seeds:", [f"{r['macro']:.4f}" for r in all_results])
print(f"Mean macro: {mean_macro:.4f} ± {std_macro:.4f}")
print('
Overall acc at best-macro epochs:', [f\"{r['acc']:.4f}\" for r in all_results])
print(f"Mean overall: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"\nPer-type accuracy (mean ± std across seeds):")
for t in SOURCE_TYPES:
    m, s = per_type_mean[t]
    print(f"  {t:6s}: {m:.4f} ± {s:.4f}")

print(f"\nBaselines:")
print(f"  Majority overall: {majority_acc:.4f}   |   majority macro: {majority_macro:.4f}")
print(f"  Random overall:   {random_acc:.4f}")

lift_macro = mean_macro - majority_macro
lift_overall = mean_acc - majority_acc
print(f"\nSourceFormer lift over majority:")
print(f"  Macro:   {lift_macro:+.4f}")
print(f"  Overall: {lift_overall:+.4f}")

best_overall_seed = SEEDS[int(np.argmax([r["macro"] for r in all_results]))]
best_ckpt_path = CKPT_DIR / f"sourceformer_k3_seed{best_overall_seed}_best.pt"

# Verdict logic — now driven by macro, with a more reasonable bar
MACRO_THRESHOLD = 0.65  # 65% balanced accuracy = clear lift over majority macro of 33%
print()
if mean_macro >= MACRO_THRESHOLD:
    print(f"✓ Macro accuracy {mean_macro:.4f} ≥ {MACRO_THRESHOLD} — PROCEED TO PHASE 5")
    print(f"  Routing is balanced across all 3 types.")
else:
    print(f"⚠ Macro accuracy {mean_macro:.4f} < {MACRO_THRESHOLD}")
    print(f"  Inspect which type still lags before proceeding.")

print(f"\nFor Phase 5 integration:")
print(f"  from sourceformer import SourceFormerK3, TYPE_TO_DATASETS, SOURCE_TYPES")
print(f"  ckpt = torch.load('{best_ckpt_path}')")
print(f"  sf = SourceFormerK3()")
print(f"  sf.load_state_dict(ckpt['state_dict'])")

print(f"\nGradient bridge: {'✓' if bridge_ok else '✗'}  (total_grad_norm={total_gnorm:.6f})")

with open("k3_summary.json", "w") as f:
    json.dump(
        to_json_safe({
            "source_types": SOURCE_TYPES,
            "type_to_datasets": TYPE_TO_DATASETS,
            "selection_criterion": "macro_per_type",
            "mean_val_macro": float(mean_macro),
            "std_val_macro": float(std_macro),
            "mean_val_acc": float(mean_acc),
            "std_val_acc": float(std_acc),
            "per_type_mean": {t: {"mean": per_type_mean[t][0], "std": per_type_mean[t][1]} for t in SOURCE_TYPES},
            "per_seed": [{"seed": s, **r} for s, r in zip(SEEDS, all_results)],
            "best_checkpoint": str(best_ckpt_path),
            "gradient_bridge_ok": bridge_ok,
            "total_grad_norm": float(total_gnorm),
            "macro_threshold_passed": bool(mean_macro >= MACRO_THRESHOLD),
            "baselines": {
                "majority_class": SOURCE_TYPES[majority_class],
                "majority_overall_acc": float(majority_acc),
                "majority_macro_acc": float(majority_macro),
                "random_acc": float(random_acc),
                "ambiguity_rate": float(ambiguity_rate),
            },
            "lift_macro": float(lift_macro),
            "lift_overall": float(lift_overall),
            "loss_function": "kl_soft_target",
            "tie_handling": "type_aggregated_argmax",
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "epochs_max": EPOCHS,
        }), # <- close to_json_safe here
        f, # <- file handle is 2nd arg to json.dump
        indent=2 # <- indent is 3rd arg to json.dump
    )
print("Saved k3_summary.json")