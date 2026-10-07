"""
script_6_joint_training.py (v2)

Patches vs v1:
  P1. Per-type alpha: KG keeps strong L_route supervision (alpha=0.7),
      text trusts L_ans more (alpha=0.4). Prevents KG collapse under
      joint training.
  P2. WeightedRandomSampler: KG upsampled 3x to match gradient frequency
      with its natural occurrence rate in the LLM's answer signal.
  P3. Save final model state unconditionally at end of each seed.
      Fixes the noisy-validation-subset checkpointing bug.
  P4. Full dev set validation (not 30-batch subset). Slower but honest.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import time
import gc
import random
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # PATCH P2
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from sourceformer import (
    SourceFormerK3, gumbel_softmax,
    SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS,
    EMBED_DIM,
)
from phase4_components import (
    RunningNormalizer, LossWeights, LossState,
    compute_l_aux, compute_combined_loss,
    gumbel_hard_pick, format_chunks, LLMConfig, ChunkDB,
)

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EPOCHS = 5
BATCH_SIZE = 8
GRAD_ACCUM = 2
TOP_K = 10
LR = 1e-4
WEIGHT_DECAY = 1e-3
GUMBEL_TAU = 1.0
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.05
VAL_EVERY = 400        # PATCH P4: less frequent so full-dev val is affordable
LOG_EVERY = 20
SEEDS = [42, 123, 2026]

# PATCH P1: per-type alpha — how much to trust L_route vs L_ans per source type
# text=0.4 (trust answer signal more), kg=0.7 (stay close to supervised routing)
ALPHA_BY_TYPE = torch.tensor([0.4, 0.5, 0.7], dtype=torch.float32)  # text, table, kg

TRAIN_FILE = "mmrag_train.json"
DEV_FILE = "mmrag_dev.json"
INDICES_DIR = Path("faiss_indices")
PHASE3_CKPT_DIR = Path("checkpoints")
PHASE4_CKPT_DIR = Path("checkpoints_phase4_v2"); PHASE4_CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("phase4_logs_v2"); LOG_DIR.mkdir(exist_ok=True)

llm_cfg = LLMConfig(model_name=LLM_MODEL, max_seq_len=2048)

# LossWeights used only for L_aux weighting — alpha handled per-type now
AUX_WEIGHT = 0.25   # L_aux share within the answer-quality component


# =============================================================================
# Soft-label construction
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
    out = []
    for item in data:
        soft = soft_target(item)
        if soft is None:
            continue
        out.append({
            "id": item.get("id", ""),
            "query": item["query"],
            "answer": item["answer"],
            "soft_label": soft,
            "hard_label": int(np.argmax(soft)),
            "relevant_chunks": item["relevant_chunks"],
        })
    return out


# =============================================================================
# PATCH P1: per-sample L_route (returns (B,) not scalar)
# =============================================================================
def compute_l_route_per_sample(logits, soft_targets):
    """KL divergence per sample. Returns (B,) tensor."""
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1)  # (B,)


def compute_combined_loss_per_type(
    l_route_per_sample,   # (B,) per-sample KL
    l_ans,                # scalar
    l_aux,                # scalar
    hard_labels,          # (B,) long tensor
    norm_route,           # RunningNormalizer
    norm_ans,             # RunningNormalizer
    norm_aux,             # RunningNormalizer
):
    """
    PATCH P1: weight L_route per sample by type-specific alpha.
    High alpha = stay close to supervised routing signal.
    Low alpha = trust L_ans gradient bridge more.
    """
    alpha = ALPHA_BY_TYPE.to(hard_labels.device)[hard_labels]  # (B,)

    # Normalize L_route per-sample then weight by alpha
    r_raw = l_route_per_sample.detach().mean().item()
    a_raw = l_ans.detach().item()
    x_raw = l_aux.detach().item()

    norm_route.update(r_raw)
    norm_ans.update(a_raw)
    norm_aux.update(x_raw)

    scale_r = max(abs(norm_route.ema) if norm_route.ema else 1.0, 1e-6)
    scale_a = max(abs(norm_ans.ema) if norm_ans.ema else 1.0, 1e-6)
    scale_x = max(abs(norm_aux.ema) if norm_aux.ema else 1.0, 1e-6)

    l_route_n = l_route_per_sample / scale_r         # (B,) normalized
    l_route_weighted = (alpha * l_route_n).mean()    # scalar, per-type weighted

    mean_alpha = alpha.mean()
    l_ans_n = l_ans / scale_a
    l_aux_n = l_aux / scale_x
    l_inner = (1.0 - AUX_WEIGHT) * l_ans_n + AUX_WEIGHT * l_aux_n
    l_total = mean_alpha * l_route_weighted + (1.0 - mean_alpha) * l_inner

    log_dict = {
        "l_route_raw": r_raw, "l_ans_raw": a_raw, "l_aux_raw": x_raw,
        "l_total": l_total.item(),
        "mean_alpha": mean_alpha.item(),
    }
    return l_total, log_dict


# =============================================================================
# Retriever (same as v1)
# =============================================================================
class MultiSourceRetriever:
    def __init__(self):
        import faiss
        self.ds_indices = {}
        self.ds_chunk_ids = {}
        for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
            print(f"  Loading {ds} index...")
            idx = faiss.read_index(str(INDICES_DIR / ds / "index.faiss"))
            ids = np.load(INDICES_DIR / ds / "chunk_ids.npy", allow_pickle=True)
            self.ds_indices[ds] = idx
            self.ds_chunk_ids[ds] = ids

    def search_all_types(self, query_embs_np, top_k):
        B = query_embs_np.shape[0]
        results = {}
        for type_name, datasets in TYPE_TO_DATASETS.items():
            all_scores, all_cids = [], []
            for ds in datasets:
                s, fids = self.ds_indices[ds].search(query_embs_np, top_k)
                cid_map = self.ds_chunk_ids[ds]
                c = np.empty((B, top_k), dtype=object)
                for r in range(B):
                    for col in range(top_k):
                        c[r, col] = cid_map[fids[r, col]]
                all_scores.append(s); all_cids.append(c)
            if len(datasets) == 1:
                results[type_name] = all_cids[0]
            else:
                merged_s = np.concatenate(all_scores, axis=1)
                merged_c = np.concatenate(all_cids, axis=1)
                top_idx = np.argsort(-merged_s, axis=1)[:, :top_k]
                merged = np.empty((B, top_k), dtype=object)
                for r in range(B):
                    for c in range(top_k):
                        merged[r, c] = merged_c[r, top_idx[r, c]]
                results[type_name] = merged
        return results


# =============================================================================
# Recall matrix
# =============================================================================
def relevant_in_type(relevant_chunks_dict, type_name):
    relevant = set()
    for cid, score in relevant_chunks_dict.items():
        if score <= 0:
            continue
        for ds in TYPE_TO_DATASETS[type_name]:
            if ds == "kg":
                if cid.startswith("m.") or cid.startswith("g."):
                    relevant.add(cid); break
            elif cid.startswith(ds + "_"):
                relevant.add(cid); break
    return relevant


def compute_recall_matrix(retrieved_by_type, items, top_k):
    B = len(items)
    recall = np.zeros((B, K), dtype=np.float32)
    for i, item in enumerate(items):
        for j, t in enumerate(SOURCE_TYPES):
            relevant = relevant_in_type(item["relevant_chunks"], t)
            if not relevant:
                continue
            top = set(retrieved_by_type[t][i].tolist())
            if top & relevant:
                recall[i, j] = 1.0
    return torch.from_numpy(recall)


# =============================================================================
# L_ans: sequential LLM NLL (memory-safe)
# =============================================================================
def compute_l_ans_sequential(llm, llm_tok, queries, contexts, answers, cfg, device):
    losses = []
    for q, ctx, ans in zip(queries, contexts, answers):
        prompt_only = cfg.chat_template(ctx, q, answer=None)
        full = cfg.chat_template(ctx, q, answer=ans)
        p_ids = llm_tok(prompt_only, add_special_tokens=False,
                        return_tensors="pt").input_ids[0]
        f_ids = llm_tok(full, add_special_tokens=False,
                        return_tensors="pt").input_ids[0]
        if f_ids.size(0) > cfg.max_seq_len:
            overflow = f_ids.size(0) - cfg.max_seq_len
            p_ids = p_ids[overflow:]
            f_ids = f_ids[overflow:]
        plen = p_ids.size(0)
        full_ids = f_ids.unsqueeze(0).to(device)
        labels = full_ids.clone()
        labels[0, :plen] = -100
        outputs = llm(input_ids=full_ids, labels=labels)
        losses.append(outputs.loss)
        del full_ids, labels, outputs
        torch.cuda.empty_cache()
    return torch.stack(losses).mean()


# =============================================================================
# LLM + BGE loading
# =============================================================================
def load_frozen_llm():
    print(f"Loading LLM: {LLM_MODEL} (4-bit, frozen)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, quantization_config=bnb, device_map={"": 0},
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    return llm, tok


def load_bge_encoder():
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


@torch.no_grad()
def encode_queries_batch(bge, bge_tok, queries):
    prefixed = [QUERY_PREFIX + q for q in queries]
    enc = bge_tok(prefixed, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
    emb = bge(**enc).last_hidden_state[:, 0]
    return F.normalize(emb.float(), p=2, dim=1)


# =============================================================================
# Dataset + PATCH P2: WeightedRandomSampler for KG upsampling
# =============================================================================
class JointTrainingDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "query": r["query"],
            "answer": r["answer"],
            "soft_label": torch.from_numpy(r["soft_label"]),
            "hard_label": r["hard_label"],
            "relevant_chunks": r["relevant_chunks"],
        }


def collate(batch):
    return {
        "query": [b["query"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "soft_label": torch.stack([b["soft_label"] for b in batch]),
        "hard_label": torch.tensor([b["hard_label"] for b in batch], dtype=torch.long),
        "relevant_chunks": [b["relevant_chunks"] for b in batch],
    }


def make_train_loader(train_records):
    """
    PATCH P2: upsample KG 3x so its gradient signal is proportional
    to its importance, not drowned out by text/table frequency.
    """
    ds = JointTrainingDataset(train_records)

    # Weights: KG (label=2) gets 3x, others get 1x
    # PATCH note: use hard_label not label (common typo)
    sample_weights = torch.tensor([
        3.0 if r["hard_label"] == SOURCE_TYPE_IDX["kg"] else 1.0
        for r in train_records
    ], dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        ds, batch_size=BATCH_SIZE,
        sampler=sampler,          # replaces shuffle=True
        collate_fn=collate,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )


# =============================================================================
# Validation — PATCH P4: full dev set, not 30-batch subset
# =============================================================================
@torch.no_grad()
def validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records):
    sf.eval()
    val_ds = JointTrainingDataset(val_records)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate)

    all_correct, all_total = 0, 0
    per_type_correct = {t: 0 for t in SOURCE_TYPES}
    per_type_total = {t: 0 for t in SOURCE_TYPES}
    all_recalls = []
    nlls = []

    # Separate normalizers for val pass (don't pollute training EMAs)
    norm_r = RunningNormalizer(); norm_a = RunningNormalizer(); norm_x = RunningNormalizer()

    for batch in val_loader:
        query_embs = encode_queries_batch(bge, bge_tok, batch["query"])
        logits, probs, one_hot = gumbel_hard_pick(sf, query_embs, tau=GUMBEL_TAU)
        q_np = query_embs.detach().cpu().numpy().astype(np.float32)
        retrieved = retriever.search_all_types(q_np, TOP_K)
        recall_mat = compute_recall_matrix(
            retrieved, [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]], TOP_K
        ).to(DEVICE)
        l_aux = compute_l_aux(probs, recall_mat)

        picked = probs.argmax(-1).cpu().numpy()
        contexts = []
        for i, p_idx in enumerate(picked):
            cids = retrieved[SOURCE_TYPES[p_idx]][i]
            chunks = [chunk_db.get(c) for c in cids]
            contexts.append(format_chunks(chunks))

        l_ans = compute_l_ans_sequential(
            llm, llm_tok, batch["query"], contexts, batch["answer"], llm_cfg, DEVICE
        )
        nlls.append(l_ans.item())

        preds = probs.argmax(-1).cpu().numpy()
        labels = batch["hard_label"].numpy()
        all_correct += int((preds == labels).sum())
        all_total += len(labels)
        for j, t in enumerate(SOURCE_TYPES):
            mask = (labels == j)
            per_type_total[t] += int(mask.sum())
            per_type_correct[t] += int(((preds == labels) & mask).sum())
        for i, p_idx in enumerate(picked):
            all_recalls.append(float(recall_mat[i, p_idx].item()))

    sf.train()
    per_type_acc = {t: per_type_correct[t] / max(per_type_total[t], 1) for t in SOURCE_TYPES}
    macro = float(np.mean(list(per_type_acc.values())))
    return {
        "acc": all_correct / max(all_total, 1),
        "per_type_acc": per_type_acc,
        "macro_acc": macro,
        "recall_at_picked_source": float(np.mean(all_recalls)),
        "loss_ans": float(np.mean(nlls)),
    }


# =============================================================================
# Training loop
# =============================================================================
def train_one_seed(seed, train_records, val_records, bge, bge_tok, llm, llm_tok,
                   retriever, chunk_db):
    print(f"\n{'='*60}\nPhase 4 v2 — seed {seed}\n{'='*60}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Init from Phase 3
    phase3_ckpt = PHASE3_CKPT_DIR / f"sourceformer_k3_seed{seed}_best.pt"
    if not phase3_ckpt.exists():
        phase3_ckpt = max(
            PHASE3_CKPT_DIR.glob("sourceformer_k3_seed*_best.pt"),
            key=lambda p: torch.load(p, map_location="cpu").get("val_macro", 0.0),
        )
        print(f"  Using fallback Phase 3 ckpt: {phase3_ckpt.name}")
    ckpt_data = torch.load(phase3_ckpt, map_location=DEVICE)
    sf = SourceFormerK3().to(DEVICE)
    sf.load_state_dict(ckpt_data["state_dict"])
    sf.train()
    print(f"  Initialized from {phase3_ckpt.name}")

    # Pre-training baseline on full dev
    print("  Pre-training validation (full dev)...")
    pre_val = validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    print(f"  PRE: acc={pre_val['acc']:.4f} macro={pre_val['macro_acc']:.4f} "
          f"recall={pre_val['recall_at_picked_source']:.4f} NLL={pre_val['loss_ans']:.4f}")

    optimizer = torch.optim.AdamW(sf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_train_loader(train_records)  # PATCH P2
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Separate normalizers for each loss term
    norm_route = RunningNormalizer()
    norm_ans = RunningNormalizer()
    norm_aux = RunningNormalizer()

    best_val_macro = pre_val["macro_acc"]
    best_ckpt_path = PHASE4_CKPT_DIR / f"phase4_seed{seed}_best.pt"
    log = {"pre": pre_val, "steps": [], "validations": []}
    optimizer_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch_idx, batch in enumerate(train_loader):
            query_embs = encode_queries_batch(bge, bge_tok, batch["query"])
            logits, probs, one_hot = gumbel_hard_pick(sf, query_embs, tau=GUMBEL_TAU)

            q_np = query_embs.detach().cpu().numpy().astype(np.float32)
            retrieved = retriever.search_all_types(q_np, TOP_K)

            recall_mat = compute_recall_matrix(
                retrieved,
                [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]],
                TOP_K
            ).to(DEVICE)
            l_aux = compute_l_aux(probs, recall_mat)

            # Per-sample L_route for per-type alpha weighting
            l_route_per_sample = compute_l_route_per_sample(
                logits, batch["soft_label"].to(DEVICE)
            )

            picked = one_hot.detach().argmax(-1).cpu().numpy()
            contexts = []
            for i, p_idx in enumerate(picked):
                cids = retrieved[SOURCE_TYPES[p_idx]][i]
                chunks = [chunk_db.get(c) for c in cids]
                contexts.append(format_chunks(chunks))

            l_ans = compute_l_ans_sequential(
                llm, llm_tok, batch["query"], contexts, batch["answer"], llm_cfg, DEVICE
            )

            # STE bridge
            ste_bridge = (one_hot.sum(-1)).mean()
            l_ans_bridged = l_ans * ste_bridge

            # PATCH P1: per-type alpha combined loss
            l_total, log_dict = compute_combined_loss_per_type(
                l_route_per_sample, l_ans_bridged, l_aux,
                batch["hard_label"].to(DEVICE),
                norm_route, norm_ans, norm_aux,
            )

            (l_total / GRAD_ACCUM).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(sf.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % LOG_EVERY == 0:
                    elapsed = time.time() - t_start
                    print(f"  Ep{epoch} step {optimizer_step}/{total_steps}  "
                          f"L={log_dict['l_total']:.3f} "
                          f"(R={log_dict['l_route_raw']:.2f} "
                          f"A={log_dict['l_ans_raw']:.2f} "
                          f"X={log_dict['l_aux_raw']:.2f}) "
                          f"α={log_dict['mean_alpha']:.2f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"t={elapsed:.0f}s")
                    log["steps"].append({
                        "epoch": epoch, "opt_step": optimizer_step, **log_dict,
                        "lr": scheduler.get_last_lr()[0],
                    })

                # PATCH P4: validate on full dev set every VAL_EVERY steps
                if optimizer_step % VAL_EVERY == 0:
                    print(f"  Full-dev validation at step {optimizer_step}...")
                    val_m = validate_full(sf, bge, bge_tok, llm, llm_tok,
                                         retriever, chunk_db, val_records)
                    print(f"    acc={val_m['acc']:.4f} macro={val_m['macro_acc']:.4f} "
                          f"recall={val_m['recall_at_picked_source']:.4f} "
                          f"NLL={val_m['loss_ans']:.4f}")
                    log["validations"].append({"opt_step": optimizer_step, **val_m})

                    if val_m["macro_acc"] > best_val_macro:
                        best_val_macro = val_m["macro_acc"]
                        torch.save({
                            "epoch": epoch, "opt_step": optimizer_step, "seed": seed,
                            "state_dict": sf.state_dict(),
                            "val_metrics": val_m,
                            "patches": ["per_type_alpha", "kg_upsampling",
                                        "full_dev_val", "final_state_save"],
                        }, best_ckpt_path)
                        print(f"    ✓ new best (macro {best_val_macro:.4f})")

    # PATCH P3: save final model state unconditionally
    final_ckpt_path = PHASE4_CKPT_DIR / f"phase4_seed{seed}_final.pt"
    final_val = validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    torch.save({
        "epoch": EPOCHS, "opt_step": total_steps, "seed": seed,
        "state_dict": sf.state_dict(),
        "val_metrics": final_val,
        "is_final_state": True,
        "patches": ["per_type_alpha", "kg_upsampling", "full_dev_val", "final_state_save"],
    }, final_ckpt_path)
    print(f"\n  Final state saved: {final_ckpt_path}")
    print(f"  Final: acc={final_val['acc']:.4f} macro={final_val['macro_acc']:.4f} "
          f"recall={final_val['recall_at_picked_source']:.4f} NLL={final_val['loss_ans']:.4f}")

    log["final"] = final_val
    log_path = LOG_DIR / f"phase4_seed{seed}_log.json"
    with open(log_path, "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        json.dump(safe(log), f, indent=2)

    return final_val, log


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    seeds_to_run = args.seeds[:1] if args.single else args.seeds

    print(f"Phase 4 v2 — patches: per-type-alpha, KG-upsampling, full-dev-val, final-state-save")
    print(f"Seeds: {seeds_to_run}")

    train_records = load_records(TRAIN_FILE)
    val_records = load_records(DEV_FILE)
    print(f"Train: {len(train_records):,}  Val: {len(val_records):,}")

    # Show upsampled class distribution for transparency
    kg_count = sum(1 for r in train_records if r["hard_label"] == SOURCE_TYPE_IDX["kg"])
    text_count = sum(1 for r in train_records if r["hard_label"] == SOURCE_TYPE_IDX["text"])
    table_count = sum(1 for r in train_records if r["hard_label"] == SOURCE_TYPE_IDX["table"])
    print(f"Train distribution (before upsampling): "
          f"text={text_count} table={table_count} kg={kg_count}")
    print(f"Effective with KG 3x upsampling: "
          f"text={text_count} table={table_count} kg={kg_count*3} "
          f"(total effective={text_count+table_count+kg_count*3})")

    chunk_db = ChunkDB("chunk_texts.db")
    retriever = MultiSourceRetriever()
    bge, bge_tok = load_bge_encoder()
    llm, llm_tok = load_frozen_llm()

    all_finals = {}
    for seed in seeds_to_run:
        final_val, log = train_one_seed(
            seed, train_records, val_records,
            bge, bge_tok, llm, llm_tok, retriever, chunk_db
        )
        all_finals[seed] = final_val
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("PHASE 4 v2 RESULTS SUMMARY")
    print("="*60)
    print(f"{'Seed':<8} {'Acc':>8} {'Macro':>8} {'R@picked':>10} {'NLL':>8}")
    for seed, m in all_finals.items():
        print(f"{seed:<8} {m['acc']:>8.4f} {m['macro_acc']:>8.4f} "
              f"{m['recall_at_picked_source']:>10.4f} {m['loss_ans']:>8.4f}")

    if len(all_finals) > 1:
        macros = [m["macro_acc"] for m in all_finals.values()]
        nlls = [m["loss_ans"] for m in all_finals.values()]
        print(f"\nMean macro: {np.mean(macros):.4f} ± {np.std(macros):.4f}")
        print(f"Mean NLL:   {np.mean(nlls):.4f} ± {np.std(nlls):.4f}")

    with open(LOG_DIR / "phase4_v2_summary.json", "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        json.dump(safe(all_finals), f, indent=2)


if __name__ == "__main__":
    main()
