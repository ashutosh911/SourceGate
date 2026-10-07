"""
script_8b_random_init.py

Tests Claim 3 from Section 7.2: can the gradient bridge learn routing
from random initialization, without any supervised pretraining?

Experiment design:
  - Initialize SourceFormer with random weights (no Phase 3 checkpoint)
  - Set alpha=0 throughout training (no L_route supervision ever)
  - Train Phase 4 with full L_ans + L_aux only
  - Compare final routing macro vs:
      * Random baseline (0.337)
      * Phase 3 supervised (0.737)
      * Phase 4 main (0.717)

Possible outcomes:
  * macro >= 0.55: STRONG result. Bridge alone learns routing from scratch.
  * macro in 0.40-0.55: Bridge has signal but weak vs supervised.
  * macro <= 0.40: Bridge cannot learn routing without initialization.
  
Either way, the answer is a real experimental result.
Run time: ~9 hours, single seed (42).
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, math, time, gc, random, warnings
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
    SOURCE_TYPES, K, DATASET_TO_TYPE, TYPE_TO_DATASETS, EMBED_DIM,
)
from phase4_components import (
    RunningNormalizer, compute_l_aux,
    gumbel_hard_pick, format_chunks, LLMConfig, ChunkDB,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Reuse most of the components from script_8_ablations.py
# Differences: no Phase 3 init, longer training, alpha=0 always
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# CRITICAL DIFFERENCE: longer training because we start from scratch
EPOCHS = 10              # vs 5 for warm-start ablation
BATCH_SIZE = 8
GRAD_ACCUM = 2
TOP_K = 10
LR = 5e-4                # higher LR than warm-start (1e-4) — random init needs it
WEIGHT_DECAY = 1e-3
GUMBEL_TAU = 1.5         # higher tau initially — soft samples for exploration
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.1        # longer warmup for random init stability
VAL_EVERY = 200          # validate more often
LOG_EVERY = 20
SEED = 42
AUX_WEIGHT = 0.25        # gamma — only L_aux + L_ans contribute (alpha=0)

TRAIN_FILE = "mmrag_train.json"
DEV_FILE = "mmrag_dev.json"
INDICES_DIR = Path("faiss_indices")
CKPT_DIR = Path("checkpoints_phase4_random_init"); CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("phase4_logs_random_init"); LOG_DIR.mkdir(exist_ok=True)

llm_cfg = LLMConfig(model_name=LLM_MODEL, max_seq_len=2048)


# Reused soft-label loading
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


# Combined loss with alpha FORCED to 0
def compute_combined_loss_random(l_ans, l_aux, norm_ans, norm_aux):
    a_raw = l_ans.detach().item()
    x_raw = l_aux.detach().item()
    norm_ans.update(a_raw)
    norm_aux.update(x_raw)
    scale_a = max(abs(norm_ans.ema or 1.0), 1e-6)
    scale_x = max(abs(norm_aux.ema or 1.0), 1e-6)
    l_ans_n = l_ans / scale_a
    l_aux_n = l_aux / scale_x
    # ALPHA = 0: no L_route at all. Combined = L_inner only
    l_total = (1.0 - AUX_WEIGHT) * l_ans_n + AUX_WEIGHT * l_aux_n
    return l_total, {"l_ans_raw": a_raw, "l_aux_raw": x_raw,
                     "l_total": l_total.item()}


# Reused retriever (same as ablation script)
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
        if score <= 0: continue
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
            if not relevant: continue
            top = set(retrieved_by_type[t][i].tolist())
            if top & relevant:
                recall[i, j] = 1.0
    return torch.from_numpy(recall)


def compute_l_ans_sequential(llm, llm_tok, queries, contexts, answers, cfg, device):
    losses = []
    for q, ctx, ans in zip(queries, contexts, answers):
        prompt_only = cfg.chat_template(ctx, q, answer=None)
        full = cfg.chat_template(ctx, q, answer=ans)
        p_ids = llm_tok(prompt_only, add_special_tokens=False, return_tensors="pt").input_ids[0]
        f_ids = llm_tok(full, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if f_ids.size(0) > cfg.max_seq_len:
            ov = f_ids.size(0) - cfg.max_seq_len
            p_ids = p_ids[ov:]; f_ids = f_ids[ov:]
        plen = p_ids.size(0)
        full_ids = f_ids.unsqueeze(0).to(device)
        labels = full_ids.clone(); labels[0, :plen] = -100
        out = llm(input_ids=full_ids, labels=labels)
        losses.append(out.loss)
        del full_ids, labels, out
        torch.cuda.empty_cache()
    return torch.stack(losses).mean()


def load_frozen_llm():
    print(f"Loading LLM: {LLM_MODEL}...")
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
def encode_queries_batch(bge, bge_tok, queries):
    prefixed = [QUERY_PREFIX + q for q in queries]
    enc = bge_tok(prefixed, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
    emb = bge(**enc).last_hidden_state[:, 0]
    return F.normalize(emb.float(), p=2, dim=1)


class JointDataset(Dataset):
    def __init__(self, records): self.records = records
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


@torch.no_grad()
def validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records):
    sf.eval()
    val_loader = DataLoader(JointDataset(val_records), batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate)
    all_correct, all_total = 0, 0
    per_type_correct = {t: 0 for t in SOURCE_TYPES}
    per_type_total = {t: 0 for t in SOURCE_TYPES}
    recalls, nlls = [], []
    for batch in val_loader:
        q_embs = encode_queries_batch(bge, bge_tok, batch["query"])
        logits, probs, _ = gumbel_hard_pick(sf, q_embs, tau=GUMBEL_TAU)
        q_np = q_embs.detach().cpu().numpy().astype(np.float32)
        retrieved = retriever.search_all_types(q_np, TOP_K)
        recall_mat = compute_recall_matrix(retrieved,
            [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]], TOP_K).to(DEVICE)
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
        all_total += len(labels)
        for j, t in enumerate(SOURCE_TYPES):
            mask = (labels == j)
            per_type_total[t] += int(mask.sum())
            per_type_correct[t] += int(((picked == labels) & mask).sum())
        for i, p_idx in enumerate(picked):
            recalls.append(float(recall_mat[i, p_idx].item()))
    sf.train()
    per_type_acc = {t: per_type_correct[t] / max(per_type_total[t], 1) for t in SOURCE_TYPES}
    macro = float(np.mean(list(per_type_acc.values())))
    return {"acc": all_correct / max(all_total, 1), "per_type_acc": per_type_acc,
            "macro_acc": macro, "recall_at_picked_source": float(np.mean(recalls)),
            "loss_ans": float(np.mean(nlls))}


def main():
    print(f"\n{'='*60}")
    print(f"BRIDGE FROM RANDOM INITIALIZATION (Claim 3)")
    print(f"{'='*60}")
    print(f"  Seed: {SEED}")
    print(f"  Init: random (NO Phase 3 checkpoint)")
    print(f"  Loss: ONLY L_ans + L_aux (alpha=0, no L_route)")
    print(f"  LR: {LR} (higher than warm-start)")
    print(f"  Epochs: {EPOCHS} (longer than warm-start)")
    print(f"  τ: {GUMBEL_TAU} (higher for exploration)\n")

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    train_records = load_records(TRAIN_FILE)
    val_records = load_records(DEV_FILE)
    print(f"  Train: {len(train_records):,}  Val: {len(val_records):,}")

    chunk_db = ChunkDB("chunk_texts.db")
    retriever = MultiSourceRetriever()
    bge, bge_tok = load_bge()
    llm, llm_tok = load_frozen_llm()

    # CRITICAL: random init, no checkpoint loaded
    sf = SourceFormerK3().to(DEVICE)
    sf.train()
    print(f"\n  SourceFormer initialized RANDOMLY (no Phase 3 ckpt)")

    # Pre-training validation — should be near random (~0.33 macro)
    print(f"\n  Pre-training validation (full dev) — expect ~random performance...")
    pre_val = validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    print(f"  PRE: acc={pre_val['acc']:.4f} macro={pre_val['macro_acc']:.4f} "
          f"recall={pre_val['recall_at_picked_source']:.4f} NLL={pre_val['loss_ans']:.4f}")
    print(f"  Per-type: {pre_val['per_type_acc']}")

    optimizer = torch.optim.AdamW(sf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(JointDataset(train_records), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate, num_workers=0,
                              pin_memory=(DEVICE == "cuda"))
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    norm_ans = RunningNormalizer()
    norm_aux = RunningNormalizer()
    best_val_macro = 0.0
    best_ckpt = CKPT_DIR / f"phase4_random_init_seed{SEED}_best.pt"
    log = {"experiment": "bridge_from_random_init", "seed": SEED,
           "pre": pre_val, "steps": [], "validations": []}
    opt_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch_idx, batch in enumerate(train_loader):
            q_embs = encode_queries_batch(bge, bge_tok, batch["query"])
            logits, probs, one_hot = gumbel_hard_pick(sf, q_embs, tau=GUMBEL_TAU)

            q_np = q_embs.detach().cpu().numpy().astype(np.float32)
            retrieved = retriever.search_all_types(q_np, TOP_K)
            recall_mat = compute_recall_matrix(retrieved,
                [{"relevant_chunks": rc} for rc in batch["relevant_chunks"]],
                TOP_K).to(DEVICE)
            l_aux = compute_l_aux(probs, recall_mat)

            picked = one_hot.detach().argmax(-1).cpu().numpy()
            contexts = []
            for i, p_idx in enumerate(picked):
                cids = retrieved[SOURCE_TYPES[p_idx]][i]
                chunks = [chunk_db.get(c) for c in cids]
                contexts.append(format_chunks(chunks))

            l_ans = compute_l_ans_sequential(llm, llm_tok, batch["query"], contexts,
                                              batch["answer"], llm_cfg, DEVICE)
            ste_bridge = (one_hot.sum(-1)).mean()
            l_ans_bridged = l_ans * ste_bridge

            l_total, log_dict = compute_combined_loss_random(
                l_ans_bridged, l_aux, norm_ans, norm_aux
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
                          f"(A={log_dict['l_ans_raw']:.2f} X={log_dict['l_aux_raw']:.2f}) "
                          f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.0f}s")
                    log["steps"].append({"epoch": epoch, "opt_step": opt_step,
                                         **log_dict, "lr": scheduler.get_last_lr()[0]})

                if opt_step % VAL_EVERY == 0:
                    print(f"  Full-dev validation at step {opt_step}...")
                    val_m = validate_full(sf, bge, bge_tok, llm, llm_tok,
                                          retriever, chunk_db, val_records)
                    print(f"    acc={val_m['acc']:.4f} macro={val_m['macro_acc']:.4f} "
                          f"recall={val_m['recall_at_picked_source']:.4f} "
                          f"NLL={val_m['loss_ans']:.4f}")
                    print(f"    per_type: {val_m['per_type_acc']}")
                    log["validations"].append({"opt_step": opt_step, **val_m})
                    if val_m["macro_acc"] > best_val_macro:
                        best_val_macro = val_m["macro_acc"]
                        torch.save({"epoch": epoch, "opt_step": opt_step, "seed": SEED,
                                    "state_dict": sf.state_dict(), "val_metrics": val_m,
                                    "experiment": "bridge_from_random_init"}, best_ckpt)
                        print(f"    ✓ new best (macro {best_val_macro:.4f})")

    # Final state
    final_ckpt = CKPT_DIR / f"phase4_random_init_seed{SEED}_final.pt"
    final_val = validate_full(sf, bge, bge_tok, llm, llm_tok, retriever, chunk_db, val_records)
    torch.save({"epoch": EPOCHS, "opt_step": total_steps, "seed": SEED,
                "state_dict": sf.state_dict(), "val_metrics": final_val,
                "experiment": "bridge_from_random_init",
                "is_final_state": True}, final_ckpt)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT — Bridge from random init")
    print(f"{'='*60}")
    print(f"  Pre-training (random init): macro={pre_val['macro_acc']:.4f}")
    print(f"  After {EPOCHS} epochs (best): macro={best_val_macro:.4f}")
    print(f"  After {EPOCHS} epochs (final): macro={final_val['macro_acc']:.4f}")
    print(f"  Final per-type: {final_val['per_type_acc']}")
    print(f"\n  For comparison:")
    print(f"    Random baseline:        0.337")
    print(f"    BGE-confidence:         0.661")
    print(f"    Phase 3 (supervised):   0.737")
    print(f"    Phase 4 main (warm):    0.717 (dev: 0.715)")
    print(f"    Phase 4 alpha=0 (warm): 0.718 (dev)")

    if final_val['macro_acc'] >= 0.55:
        print(f"\n  ✓ STRONG result: bridge alone learns routing from scratch")
    elif final_val['macro_acc'] >= 0.40:
        print(f"\n  ≈ Partial result: bridge has signal but weaker than supervised")
    else:
        print(f"\n  ✗ Weak result: bridge alone insufficient without supervised init")

    log["final"] = final_val
    log["best_macro"] = best_val_macro
    with open(LOG_DIR / f"phase4_random_init_seed{SEED}_log.json", "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        json.dump(safe(log), f, indent=2)
    print(f"\n  Log saved to {LOG_DIR}/phase4_random_init_seed{SEED}_log.json")


if __name__ == "__main__":
    main()
