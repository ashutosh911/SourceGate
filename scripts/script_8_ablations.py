"""
script_8_ablations.py

Ablation experiments for Phase 4. Reuses the K=3 training pipeline with
specific loss-component overrides.

Two ablations:

  --ablation alpha0   : Set α=0. Bridge-only — tests if L_ans gradient
                        through Gumbel-STE alone improves routing without
                        any supervised L_route signal. Initialized from
                        Phase 3 checkpoint (1b setup).

  --ablation no_aux   : Set γ=0. Removes auxiliary retrieval recall loss.
                        Tests whether L_aux is necessary or if L_ans alone
                        provides enough signal.

Outputs to checkpoints_phase4_ablation/ and phase4_logs_ablation/.
Each ablation uses seed 42 by default for direct comparison with seed 42 main run.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
)

from sourceformer import (
    SourceFormerK3, gumbel_softmax,
    SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS,
    EMBED_DIM,
)
from phase4_components import (
    RunningNormalizer, compute_l_aux,
    gumbel_hard_pick, format_chunks, LLMConfig, ChunkDB,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Config — same as main script_6 unless noted
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
VAL_EVERY = 400
LOG_EVERY = 20

TRAIN_FILE = "mmrag_train.json"
DEV_FILE = "mmrag_dev.json"
INDICES_DIR = Path("faiss_indices")
PHASE3_CKPT_DIR = Path("checkpoints")
ABLATION_CKPT_DIR = Path("checkpoints_phase4_ablation"); ABLATION_CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("phase4_logs_ablation"); LOG_DIR.mkdir(exist_ok=True)

llm_cfg = LLMConfig(model_name=LLM_MODEL, max_seq_len=2048)


# =============================================================================
# Soft-label loading
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
# Per-sample L_route
# =============================================================================
def compute_l_route_per_sample(logits, soft_targets):
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1)


# =============================================================================
# Ablation-aware combined loss
# =============================================================================
def compute_combined_loss_ablation(
    l_route_per_sample, l_ans, l_aux,
    norm_route, norm_ans, norm_aux,
    alpha=0.5, gamma=0.25, ablation=None,
):
    """
    Hierarchical: L = α·L_route + (1-α)·[(1-γ)·L_ans + γ·L_aux]

    ablation:
      "alpha0":  α=0  → L_route ignored, only L_ans + L_aux drive routing
      "no_aux":  γ=0  → L_aux disabled, equivalent to α·L_route + (1-α)·L_ans
      None:      standard config (α=0.5, γ=0.25)
    """
    if ablation == "alpha0":
        alpha = 0.0
    elif ablation == "no_aux":
        gamma = 0.0

    r_raw = l_route_per_sample.detach().mean().item()
    a_raw = l_ans.detach().item()
    x_raw = l_aux.detach().item()

    norm_route.update(r_raw)
    norm_ans.update(a_raw)
    norm_aux.update(x_raw)

    scale_r = max(abs(norm_route.ema or 1.0), 1e-6)
    scale_a = max(abs(norm_ans.ema   or 1.0), 1e-6)
    scale_x = max(abs(norm_aux.ema   or 1.0), 1e-6)

    l_route_n = (l_route_per_sample / scale_r).mean()
    l_ans_n   = l_ans / scale_a
    l_aux_n   = l_aux / scale_x

    l_inner = (1.0 - gamma) * l_ans_n + gamma * l_aux_n
    l_total = alpha * l_route_n + (1.0 - alpha) * l_inner

    log_dict = {
        "l_route_raw": r_raw, "l_ans_raw": a_raw, "l_aux_raw": x_raw,
        "l_total": l_total.item(),
        "alpha_used": alpha, "gamma_used": gamma,
    }
    return l_total, log_dict


# =============================================================================
# Reused components from main script (retriever, recall, LLM loss)
# =============================================================================
class MultiSourceRetriever:
    def __init__(self):
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


def load_frozen_llm():
    print(f"Loading LLM: {LLM_MODEL}...")
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
# Dataset
# =============================================================================
class JointDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        return {"query": r["query"], "answer": r["answer"],
                "soft_label": torch.from_numpy(r["soft_label"]),
                "hard_label": r["hard_label"],
                "relevant_chunks": r["relevant_chunks"]}


def collate(batch):
    return {"query": [b["query"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "soft_label": torch.stack([b["soft_label"] for b in batch]),
            "hard_label": torch.tensor([b["hard_label"] for b in batch], dtype=torch.long),
            "relevant_chunks": [b["relevant_chunks"] for b in batch]}


# =============================================================================
# Validation (full dev)
# =============================================================================
@torch.no_grad()
def validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records):
    sf.eval()
    val_ds = JointDataset(val_records)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    all_correct, all_total = 0, 0
    per_type_correct = {t: 0 for t in SOURCE_TYPES}
    per_type_total   = {t: 0 for t in SOURCE_TYPES}
    recalls, nlls = [], []

    for batch in val_loader:
        q_embs = encode_queries_batch(bge, bge_tok, batch["query"])
        logits, probs, _ = gumbel_hard_pick(sf, q_embs, tau=GUMBEL_TAU)
        q_np = q_embs.detach().cpu().numpy().astype(np.float32)
        retrieved = retriever.search_all_types(q_np, TOP_K)
        recall_mat = compute_recall_matrix(
            retrieved, [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]], TOP_K
        ).to(DEVICE)

        picked = probs.argmax(-1).cpu().numpy()
        contexts = []
        for i, p_idx in enumerate(picked):
            cids = retrieved[SOURCE_TYPES[p_idx]][i]
            chunks = [chunk_db.get(c) for c in cids]
            contexts.append(format_chunks(chunks))

        l_ans = compute_l_ans_sequential(llm, llm_tok, batch["query"], contexts,
                                          batch["answer"], llm_cfg, DEVICE)
        nlls.append(l_ans.item())

        labels = batch["hard_label"].numpy()
        all_correct += int((picked == labels).sum())
        all_total   += len(labels)
        for j, t in enumerate(SOURCE_TYPES):
            mask = (labels == j)
            per_type_total[t]   += int(mask.sum())
            per_type_correct[t] += int(((picked == labels) & mask).sum())
        for i, p_idx in enumerate(picked):
            recalls.append(float(recall_mat[i, p_idx].item()))

    sf.train()
    per_type_acc = {t: per_type_correct[t] / max(per_type_total[t], 1) for t in SOURCE_TYPES}
    macro = float(np.mean(list(per_type_acc.values())))
    return {"acc": all_correct / max(all_total, 1), "per_type_acc": per_type_acc,
            "macro_acc": macro, "recall_at_picked_source": float(np.mean(recalls)),
            "loss_ans": float(np.mean(nlls))}


# =============================================================================
# Train one ablation
# =============================================================================
def train_ablation(seed, ablation, train_records, val_records,
                   bge, bge_tok, llm, llm_tok, retriever, chunk_db):
    print(f"\n{'='*60}\nABLATION: {ablation}  |  seed {seed}\n{'='*60}")
    if ablation == "alpha0":
        print("  α=0: NO supervised L_route signal during training.")
        print("       Routing learned ONLY through gradient bridge (L_ans + L_aux).")
        print("       Initialized from Phase 3 checkpoint (1b setup).")
    elif ablation == "no_aux":
        print("  γ=0: NO auxiliary retrieval recall loss.")
        print("       Combined loss = 0.5·L_route + 0.5·L_ans only.")
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Init from Phase 3 checkpoint (same as main run)
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
    print(f"  Initialized from {phase3_ckpt.name}, val_macro={ckpt_data.get('val_macro'):.4f}")

    # Pre-eval on full dev
    print("  Pre-training validation (full dev)...")
    pre_val = validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    print(f"  PRE: acc={pre_val['acc']:.4f} macro={pre_val['macro_acc']:.4f} "
          f"recall={pre_val['recall_at_picked_source']:.4f} NLL={pre_val['loss_ans']:.4f}")

    optimizer = torch.optim.AdamW(sf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_ds = JointDataset(train_records)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=0,
                              pin_memory=(DEVICE == "cuda"))
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    norm_route = RunningNormalizer()
    norm_ans   = RunningNormalizer()
    norm_aux   = RunningNormalizer()
    best_val_macro = pre_val["macro_acc"]
    best_ckpt_path = ABLATION_CKPT_DIR / f"phase4_{ablation}_seed{seed}_best.pt"
    log = {"ablation": ablation, "seed": seed, "pre": pre_val,
           "steps": [], "validations": []}
    optimizer_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch_idx, batch in enumerate(train_loader):
            q_embs = encode_queries_batch(bge, bge_tok, batch["query"])
            logits, probs, one_hot = gumbel_hard_pick(sf, q_embs, tau=GUMBEL_TAU)

            q_np = q_embs.detach().cpu().numpy().astype(np.float32)
            retrieved = retriever.search_all_types(q_np, TOP_K)
            recall_mat = compute_recall_matrix(
                retrieved,
                [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]],
                TOP_K
            ).to(DEVICE)
            l_aux = compute_l_aux(probs, recall_mat)

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
            ste_bridge = (one_hot.sum(-1)).mean()
            l_ans_bridged = l_ans * ste_bridge

            l_total, log_dict = compute_combined_loss_ablation(
                l_route_per_sample, l_ans_bridged, l_aux,
                norm_route, norm_ans, norm_aux,
                ablation=ablation,
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
                          f"α={log_dict['alpha_used']:.1f} "
                          f"γ={log_dict['gamma_used']:.2f} "
                          f"t={elapsed:.0f}s")
                    log["steps"].append({
                        "epoch": epoch, "opt_step": optimizer_step, **log_dict,
                        "lr": scheduler.get_last_lr()[0],
                    })

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
                        torch.save({"epoch": epoch, "opt_step": optimizer_step,
                                    "seed": seed, "ablation": ablation,
                                    "state_dict": sf.state_dict(),
                                    "val_metrics": val_m}, best_ckpt_path)
                        print(f"    ✓ new best (macro {best_val_macro:.4f})")

    # Save final state unconditionally
    final_ckpt_path = ABLATION_CKPT_DIR / f"phase4_{ablation}_seed{seed}_final.pt"
    final_val = validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    torch.save({"epoch": EPOCHS, "opt_step": total_steps, "seed": seed,
                "ablation": ablation, "state_dict": sf.state_dict(),
                "val_metrics": final_val, "is_final_state": True}, final_ckpt_path)
    print(f"\n  Final state saved → {final_ckpt_path.name}")
    print(f"  Final: acc={final_val['acc']:.4f} macro={final_val['macro_acc']:.4f} "
          f"recall={final_val['recall_at_picked_source']:.4f} NLL={final_val['loss_ans']:.4f}")
    log["final"] = final_val

    log_path = LOG_DIR / f"phase4_{ablation}_seed{seed}_log.json"
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
    parser.add_argument("--ablations", nargs="+",
                        choices=["alpha0", "no_aux"],
                        default=["alpha0", "no_aux"],
                        help="Which ablations to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed (default: 42 for direct comparison with main run)")
    args = parser.parse_args()

    print(f"Phase 4 ablation experiments")
    print(f"  Ablations: {args.ablations}")
    print(f"  Seed: {args.seed}")

    train_records = load_records(TRAIN_FILE)
    val_records   = load_records(DEV_FILE)
    print(f"  Train: {len(train_records):,}  Val: {len(val_records):,}")

    chunk_db  = ChunkDB("chunk_texts.db")
    retriever = MultiSourceRetriever()
    bge, bge_tok = load_bge_encoder()
    llm, llm_tok = load_frozen_llm()

    all_results = {}
    for ablation in args.ablations:
        final_val, _ = train_ablation(
            args.seed, ablation, train_records, val_records,
            bge, bge_tok, llm, llm_tok, retriever, chunk_db
        )
        all_results[ablation] = final_val
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*65)
    print(f"ABLATION RESULTS (seed {args.seed})")
    print("="*65)
    print(f"{'Ablation':<15} {'Acc':>8} {'Macro':>8} {'R@picked':>10} {'NLL':>8}")
    for ablation, m in all_results.items():
        print(f"{ablation:<15} {m['acc']:>8.4f} {m['macro_acc']:>8.4f} "
              f"{m['recall_at_picked_source']:>10.4f} {m['loss_ans']:>8.4f}")

    # Reference: main Phase 4 seed 42 numbers (for direct comparison)
    print(f"\nReference (main Phase 4 seed 42):")
    print(f"{'main':<15} {'0.7522':>8} {'0.7147':>8} {'0.7572':>10} {'3.8083':>8}")

    with open(LOG_DIR / f"ablation_summary_seed{args.seed}.json", "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        json.dump(safe(all_results), f, indent=2)


if __name__ == "__main__":
    main()
