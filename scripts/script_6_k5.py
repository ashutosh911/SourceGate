"""
script_6_k5.py: Phase 4 joint training for K=5 routing.
All four patches applied. Seeds 7 and 99.

Key differences from K=3 script_6_v2:
  - K=5 routing (nq, triviaqa, ott, tat, kg)
  - Retrieval: one FAISS index per source, no merging needed
  - Per-type alpha: [0.4, 0.4, 0.5, 0.6, 0.7] for [nq, tri, ott, tat, kg]
  - Upsampling: TAT 2.5x, KG 2.0x, OTT 1.5x
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, math, time, gc, random, argparse, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import faiss
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig)

from sourceformer import (
    SourceFormerK5, gumbel_softmax,
    SOURCE_TYPES_K5, SOURCE_TYPE_IDX_K5, K5,
    TYPE_TO_DATASET_K5, EMBED_DIM,
)
from phase4_components import (
    RunningNormalizer, compute_l_aux,
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
VAL_EVERY = 400
LOG_EVERY = 20
SEEDS = [7, 99]
AUX_WEIGHT = 0.25

# Per-type alpha for K=5: [nq, triviaqa, ott, tat, kg]
# tat and kg get higher alpha to preserve routing supervision on small/hard sources
ALPHA_BY_TYPE_K5 = torch.tensor([0.4, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)

# Upsampling weights for K=5
UPSAMPLE_WEIGHTS_K5 = {
    SOURCE_TYPE_IDX_K5["nq"]:       1.0,
    SOURCE_TYPE_IDX_K5["triviaqa"]: 1.0,
    SOURCE_TYPE_IDX_K5["ott"]:      1.5,
    SOURCE_TYPE_IDX_K5["tat"]:      2.5,
    SOURCE_TYPE_IDX_K5["kg"]:       2.0,
}

TRAIN_FILE = "mmrag_train.json"
DEV_FILE = "mmrag_dev.json"
INDICES_DIR = Path("faiss_indices")
PHASE3_K5_CKPT_DIR = Path("checkpoints")
PHASE4_K5_CKPT_DIR = Path("checkpoints_phase4_k5"); PHASE4_K5_CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("phase4_logs_k5"); LOG_DIR.mkdir(exist_ok=True)

llm_cfg = LLMConfig(model_name=LLM_MODEL, max_seq_len=2048)


# =============================================================================
# Soft labels for K=5
# =============================================================================
def soft_target_k5(item):
    scores = item["dataset_score"]
    vec = np.array([float(scores.get(t, 0.0)) for t in SOURCE_TYPES_K5], dtype=np.float32)
    total = vec.sum()
    if total == 0:
        return None
    return vec / total


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    out = []
    for item in data:
        soft = soft_target_k5(item)
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
# Per-type alpha + normalized combined loss for K=5
# =============================================================================
def compute_l_route_per_sample_k5(logits, soft_targets):
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1)  # (B,)


def compute_combined_loss_k5(l_route_per_sample, l_ans, l_aux,
                              hard_labels, norm_route, norm_ans, norm_aux):
    alpha = ALPHA_BY_TYPE_K5.to(hard_labels.device)[hard_labels]
    r_raw = l_route_per_sample.detach().mean().item()
    a_raw = l_ans.detach().item()
    x_raw = l_aux.detach().item()
    norm_route.update(r_raw); norm_ans.update(a_raw); norm_aux.update(x_raw)
    scale_r = max(abs(norm_route.ema or 1.0), 1e-6)
    scale_a = max(abs(norm_ans.ema or 1.0), 1e-6)
    scale_x = max(abs(norm_aux.ema or 1.0), 1e-6)
    l_route_n = l_route_per_sample / scale_r
    l_route_weighted = (alpha * l_route_n).mean()
    mean_alpha = alpha.mean()
    l_inner = (1.0 - AUX_WEIGHT) * (l_ans / scale_a) + AUX_WEIGHT * (l_aux / scale_x)
    l_total = mean_alpha * l_route_weighted + (1.0 - mean_alpha) * l_inner
    return l_total, {"l_route_raw": r_raw, "l_ans_raw": a_raw, "l_aux_raw": x_raw,
                     "l_total": l_total.item(), "mean_alpha": mean_alpha.item()}


# =============================================================================
# K=5 retriever — one index per source, NO merging
# =============================================================================
class K5Retriever:
    """
    K=5 retrieval is simpler than K=3: each source type maps to exactly
    one FAISS index. No inter-dataset merging needed.
    """
    def __init__(self):
        self.indices = {}
        self.chunk_ids = {}
        for src in SOURCE_TYPES_K5:
            ds = TYPE_TO_DATASET_K5[src]
            print(f"  Loading {src} ({ds}) index...")
            idx = faiss.read_index(str(INDICES_DIR / ds / "index.faiss"))
            ids = np.load(INDICES_DIR / ds / "chunk_ids.npy", allow_pickle=True)
            self.indices[src] = idx
            self.chunk_ids[src] = ids
            print(f"    {src}: {idx.ntotal:,} vectors")

    def search_all(self, query_embs_np, top_k):
        """Returns dict {source_type: (B, top_k) chunk_ids}."""
        B = query_embs_np.shape[0]
        results = {}
        for src in SOURCE_TYPES_K5:
            scores, fids = self.indices[src].search(query_embs_np, top_k)
            cid_map = self.chunk_ids[src]
            cids = np.empty((B, top_k), dtype=object)
            for r in range(B):
                for c in range(top_k):
                    cids[r, c] = cid_map[fids[r, c]]
            results[src] = cids
        return results

    def search_one(self, query_embs_np, src, top_k):
        B = query_embs_np.shape[0]
        scores, fids = self.indices[src].search(query_embs_np, top_k)
        cid_map = self.chunk_ids[src]
        cids = np.empty((B, top_k), dtype=object)
        for r in range(B):
            for c in range(top_k):
                cids[r, c] = cid_map[fids[r, c]]
        return cids


# =============================================================================
# Recall matrix for K=5
# =============================================================================
def relevant_for_source(relevant_chunks_dict, src):
    """K=5: one dataset per source, prefix check is direct."""
    relevant = set()
    for cid, score in relevant_chunks_dict.items():
        if score <= 0:
            continue
        if src == "kg":
            if cid.startswith("m.") or cid.startswith("g."):
                relevant.add(cid)
        elif cid.startswith(src + "_"):
            relevant.add(cid)
    return relevant


def compute_recall_matrix_k5(retrieved_all, items, top_k):
    B = len(items)
    recall = np.zeros((B, K5), dtype=np.float32)
    for i, item in enumerate(items):
        for j, src in enumerate(SOURCE_TYPES_K5):
            relevant = relevant_for_source(item["relevant_chunks"], src)
            if not relevant:
                continue
            top = set(retrieved_all[src][i].tolist())
            if top & relevant:
                recall[i, j] = 1.0
    return torch.from_numpy(recall)


# =============================================================================
# LLM + BGE
# =============================================================================
def load_frozen_llm():
    print(f"Loading {LLM_MODEL}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=bnb,
                                                device_map={"": 0})
    llm.eval()
    for p in llm.parameters(): p.requires_grad = False
    return llm, tok


def load_bge():
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad = False
    return model, tok


@torch.no_grad()
def encode_batch(bge, bge_tok, queries):
    prefixed = [QUERY_PREFIX + q for q in queries]
    enc = bge_tok(prefixed, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
    emb = bge(**enc).last_hidden_state[:, 0]
    return F.normalize(emb.float(), p=2, dim=1)


def compute_l_ans_sequential(llm, tok, queries, contexts, answers, cfg, device):
    losses = []
    for q, ctx, ans in zip(queries, contexts, answers):
        p = tok(cfg.chat_template(ctx, q), add_special_tokens=False,
                return_tensors="pt").input_ids[0]
        f = tok(cfg.chat_template(ctx, q, ans), add_special_tokens=False,
                return_tensors="pt").input_ids[0]
        if f.size(0) > cfg.max_seq_len:
            ov = f.size(0) - cfg.max_seq_len
            p = p[ov:]; f = f[ov:]
        ids = f.unsqueeze(0).to(device)
        lbl = ids.clone(); lbl[0, :p.size(0)] = -100
        out = llm(input_ids=ids, labels=lbl)
        losses.append(out.loss)
        del ids, lbl, out; torch.cuda.empty_cache()
    return torch.stack(losses).mean()


# =============================================================================
# Dataset + upsampling
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


def make_loader(records, shuffle=False):
    ds = JointDataset(records)
    if shuffle:
        w = torch.tensor([UPSAMPLE_WEIGHTS_K5[r["hard_label"]] for r in records],
                         dtype=torch.float32)
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=collate, num_workers=0, pin_memory=(DEVICE=="cuda"))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate, num_workers=0, pin_memory=(DEVICE=="cuda"))


# =============================================================================
# Full-dev validation
# =============================================================================
@torch.no_grad()
def validate_full_k5(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records):
    sf.eval()
    loader = make_loader(val_records, shuffle=False)
    all_correct, all_total = 0, 0
    per_type_correct = {t: 0 for t in SOURCE_TYPES_K5}
    per_type_total = {t: 0 for t in SOURCE_TYPES_K5}
    recalls, nlls = [], []

    for batch in loader:
        q_embs = encode_batch(bge, bge_tok, batch["query"])
        logits = sf(q_embs)
        probs = F.softmax(logits, dim=-1)
        picked = probs.argmax(-1).cpu().numpy()
        q_np = q_embs.detach().cpu().numpy().astype(np.float32)
        retrieved = retriever.search_all(q_np, TOP_K)
        recall_mat = compute_recall_matrix_k5(
            retrieved, [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]], TOP_K
        ).to(DEVICE)
        l_aux = compute_l_aux(probs, recall_mat)
        contexts = []
        for i, p_idx in enumerate(picked):
            cids = retrieved[SOURCE_TYPES_K5[p_idx]][i]
            chunks = [chunk_db.get(c) for c in cids]
            contexts.append(format_chunks(chunks))
        l_ans = compute_l_ans_sequential(llm, llm_tok, batch["query"], contexts,
                                          batch["answer"], llm_cfg, DEVICE)
        nlls.append(l_ans.item())
        labels = batch["hard_label"].numpy()
        all_correct += int((picked == labels).sum())
        all_total += len(labels)
        for j, t in enumerate(SOURCE_TYPES_K5):
            mask = (labels == j)
            per_type_total[t] += int(mask.sum())
            per_type_correct[t] += int(((picked == labels) & mask).sum())
        for i, p_idx in enumerate(picked):
            recalls.append(float(recall_mat[i, p_idx].item()))

    sf.train()
    per_type_acc = {t: per_type_correct[t] / max(per_type_total[t], 1) for t in SOURCE_TYPES_K5}
    macro = float(np.mean(list(per_type_acc.values())))
    return {"acc": all_correct / max(all_total, 1), "per_type_acc": per_type_acc,
            "macro_acc": macro, "recall_at_picked_source": float(np.mean(recalls)),
            "loss_ans": float(np.mean(nlls))}


# =============================================================================
# Train one seed
# =============================================================================
def train_one_seed(seed, train_records, val_records, bge, bge_tok, llm, llm_tok,
                   retriever, chunk_db):
    print(f"\n{'='*60}\nPhase 4 K=5 — seed {seed}\n{'='*60}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    ckpt_path = PHASE3_K5_CKPT_DIR / f"sourceformer_k5_seed{seed}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Phase 3 K=5 checkpoint not found: {ckpt_path}\n"
            f"Run script_5b_k5.py first."
        )
    ckpt_data = torch.load(ckpt_path, map_location=DEVICE)
    sf = SourceFormerK5().to(DEVICE)
    sf.load_state_dict(ckpt_data["state_dict"])
    sf.train()
    print(f"  Loaded Phase 3 K=5 seed {seed}, val_macro={ckpt_data.get('val_macro', '?')}")

    print("  Pre-training validation (full dev)...")
    pre = validate_full_k5(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    print(f"  PRE: acc={pre['acc']:.4f} macro={pre['macro_acc']:.4f} "
          f"recall={pre['recall_at_picked_source']:.4f} NLL={pre['loss_ans']:.4f}")

    optimizer = torch.optim.AdamW(sf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train_records, shuffle=True)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    norm_route = RunningNormalizer()
    norm_ans = RunningNormalizer()
    norm_aux = RunningNormalizer()
    best_val_macro = pre["macro_acc"]
    best_ckpt = PHASE4_K5_CKPT_DIR / f"phase4_k5_seed{seed}_best.pt"
    log = {"pre": pre, "steps": [], "validations": []}
    opt_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch_idx, batch in enumerate(train_loader):
            q_embs = encode_batch(bge, bge_tok, batch["query"])
            logits = sf(q_embs)
            probs = F.softmax(logits, dim=-1)
            one_hot = gumbel_softmax(logits, tau=GUMBEL_TAU, hard=True)

            q_np = q_embs.detach().cpu().numpy().astype(np.float32)
            retrieved = retriever.search_all(q_np, TOP_K)

            recall_mat = compute_recall_matrix_k5(
                retrieved,
                [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]],
                TOP_K
            ).to(DEVICE)
            l_aux = compute_l_aux(probs, recall_mat)

            l_route_per_sample = compute_l_route_per_sample_k5(
                logits, batch["soft_label"].to(DEVICE)
            )

            picked = one_hot.detach().argmax(-1).cpu().numpy()
            contexts = []
            for i, p_idx in enumerate(picked):
                cids = retrieved[SOURCE_TYPES_K5[p_idx]][i]
                chunks = [chunk_db.get(c) for c in cids]
                contexts.append(format_chunks(chunks))

            l_ans = compute_l_ans_sequential(
                llm, llm_tok, batch["query"], contexts, batch["answer"], llm_cfg, DEVICE
            )
            ste_bridge = (one_hot.sum(-1)).mean()
            l_ans_bridged = l_ans * ste_bridge

            l_total, log_dict = compute_combined_loss_k5(
                l_route_per_sample, l_ans_bridged, l_aux,
                batch["hard_label"].to(DEVICE),
                norm_route, norm_ans, norm_aux,
            )

            (l_total / GRAD_ACCUM).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(sf.parameters(), GRAD_CLIP)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                opt_step += 1

                if opt_step % LOG_EVERY == 0:
                    elapsed = time.time() - t_start
                    print(f"  Ep{epoch} step {opt_step}/{total_steps}  "
                          f"L={log_dict['l_total']:.3f} "
                          f"(R={log_dict['l_route_raw']:.2f} "
                          f"A={log_dict['l_ans_raw']:.2f} "
                          f"X={log_dict['l_aux_raw']:.2f}) "
                          f"α={log_dict['mean_alpha']:.2f} t={elapsed:.0f}s")
                    log["steps"].append({"epoch": epoch, "opt_step": opt_step,
                                         **log_dict, "lr": scheduler.get_last_lr()[0]})

                if opt_step % VAL_EVERY == 0:
                    print(f"  Full-dev validation at step {opt_step}...")
                    val_m = validate_full_k5(sf, bge, bge_tok, llm, llm_tok,
                                             retriever, chunk_db, val_records)
                    print(f"    acc={val_m['acc']:.4f} macro={val_m['macro_acc']:.4f} "
                          f"recall={val_m['recall_at_picked_source']:.4f} "
                          f"NLL={val_m['loss_ans']:.4f}")
                    log["validations"].append({"opt_step": opt_step, **val_m})
                    if val_m["macro_acc"] > best_val_macro:
                        best_val_macro = val_m["macro_acc"]
                        torch.save({"epoch": epoch, "opt_step": opt_step, "seed": seed,
                                    "state_dict": sf.state_dict(), "val_metrics": val_m,
                                    "k": K5, "source_types": SOURCE_TYPES_K5},
                                   best_ckpt)
                        print(f"    ✓ new best (macro {best_val_macro:.4f})")

    # Save final state unconditionally
    final_ckpt = PHASE4_K5_CKPT_DIR / f"phase4_k5_seed{seed}_final.pt"
    final_val = validate_full_k5(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    torch.save({"epoch": EPOCHS, "opt_step": total_steps, "seed": seed,
                "state_dict": sf.state_dict(), "val_metrics": final_val,
                "is_final_state": True, "k": K5, "source_types": SOURCE_TYPES_K5},
               final_ckpt)
    print(f"\n  Final state saved → {final_ckpt}")
    print(f"  Final: acc={final_val['acc']:.4f} macro={final_val['macro_acc']:.4f} "
          f"recall={final_val['recall_at_picked_source']:.4f} NLL={final_val['loss_ans']:.4f}")
    log["final"] = final_val

    with open(LOG_DIR / f"phase4_k5_seed{seed}_log.json", "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        json.dump(safe(log), f, indent=2)

    return final_val


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    seeds_to_run = args.seeds[:1] if args.single else args.seeds

    print(f"Phase 4 K=5 — seeds {seeds_to_run}, all patches active")
    train_records = load_records(TRAIN_FILE)
    val_records = load_records(DEV_FILE)
    print(f"Train: {len(train_records):,}  Val: {len(val_records):,}")

    chunk_db = ChunkDB("chunk_texts.db")
    retriever = K5Retriever()
    bge, bge_tok = load_bge()
    llm, llm_tok = load_frozen_llm()

    all_finals = {}
    for seed in seeds_to_run:
        final = train_one_seed(seed, train_records, val_records,
                               bge, bge_tok, llm, llm_tok, retriever, chunk_db)
        all_finals[seed] = final
        gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("K=5 PHASE 4 RESULTS")
    print("="*60)
    print(f"{'Seed':<8} {'Acc':>8} {'Macro':>8} {'R@picked':>10} {'NLL':>8}")
    for seed, m in all_finals.items():
        print(f"{seed:<8} {m['acc']:>8.4f} {m['macro_acc']:>8.4f} "
              f"{m['recall_at_picked_source']:>10.4f} {m['loss_ans']:>8.4f}")

    with open(LOG_DIR / "phase4_k5_summary.json", "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        json.dump(safe(all_finals), f, indent=2)


if __name__ == "__main__":
    main()