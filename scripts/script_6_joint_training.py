"""
Script 6: Joint training — Phase 5.

Loads the pre-trained SourceFormer from Script 5 and couples it to
AnswerFormer (Llama 3.1 8B Instruct, 4-bit, LoRA fine-tuned).

Joint loss:
    L_total = α · L_route  +  (1−α) · L_ans
    α = 0.5  (balanced, matching your e-commerce paper)

Gradient flow:
    L_route → SourceFormer                        (direct supervision)
    L_ans   → AnswerFormer (LoRA) → routing weights → SourceFormer
              (through Gumbel-Softmax straight-through estimator)

Key design choices:
  - Separate learning rates: SourceFormer 1e-3, LoRA adapters 3e-4.
  - Gumbel temperature τ anneals linearly: 1.0 → 0.1 over training.
  - Gradient clipping at 1.0 (applied to each param group separately).
  - 3 seeds (42, 123, 2026), report mean ± std.
  - Early stopping on validation EM with patience=5.

INPUTS:
  - checkpoints/sourceformer_{variant}_seed{seed}_best.pt  (from script_5)
  - mmrag_train.json / mmrag_dev.json
  - faiss_indices/  (for retrieval during training)
  - query_emb_cache/ (BGE query embeddings, optional — will re-encode if missing)

OUTPUTS:
  - checkpoints/joint_seed{seed}_best.pt
  - joint_training_log.json
"""

import json
import time
import gc
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForCausalLM, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from sourceformer import (
    build_sourceformer, gumbel_softmax,
    SOURCES, SOURCE_TO_IDX, K, EMBED_DIM,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", ("Joint training requires a GPU. "
                          "AnswerFormer (Llama 4-bit) needs ≥10GB VRAM.")

BGE_NAME      = "BAAI/bge-base-en-v1.5"
LLM_NAME      = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "
MAX_QUERY_LEN = 512
MAX_GEN_LEN   = 64       # max answer tokens
MAX_CTX_LEN   = 1024    # query + retrieved context max tokens

# Retrieval
TOP_K         = 5        # retrieve top-5 chunks per source
INDICES_DIR   = Path("faiss_indices")
CHUNKS_DIR    = Path("chunks_by_source")   # original .jsonl files for text lookup

# Training
EPOCHS        = 20
LR_SF         = 1e-3     # SourceFormer
LR_LORA       = 3e-4     # LoRA adapters
WEIGHT_DECAY  = 1e-4
ALPHA         = 0.5      # joint loss weight
BATCH_SIZE    = 8        # limited by VRAM
GRAD_ACCUM    = 4        # effective batch = 32
TAU_START     = 1.0
TAU_END       = 0.1
PATIENCE      = 5
GRAD_CLIP     = 1.0
SEEDS         = [42, 123, 2026]
SF_VARIANT    = "mlp"    # set to "transformer" to use that ablation

# LoRA config
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "v_proj"]   # target modules in Llama

CKPT_DIR      = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
EMB_CACHE     = Path("query_emb_cache")

print(f"Device: {DEVICE}")
print(f"α = {ALPHA}  |  τ: {TAU_START} → {TAU_END}  |  SF_LR={LR_SF}  |  LoRA_LR={LR_LORA}")


# ---------------------------------------------------------------------------
# Chunk store — maps chunk_id → text for retrieved evidence
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    Lazy-loads chunk text from .jsonl files.
    Keeps a per-source dict in RAM (typically <1GB total).
    """
    def __init__(self, chunks_dir: Path):
        self.chunks_dir = chunks_dir
        self._stores: dict[str, dict] = {}

    def _load_source(self, src: str):
        path = self.chunks_dir / f"{src}.jsonl"
        store = {}
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                store[obj["id"]] = obj["text"]
        self._stores[src] = store
        print(f"  ChunkStore: loaded {len(store):,} chunks for {src}")

    def get(self, src: str, chunk_id: str) -> str:
        if src not in self._stores:
            self._load_source(src)
        return self._stores[src].get(chunk_id, "")


chunk_store = ChunkStore(CHUNKS_DIR)


# ---------------------------------------------------------------------------
# Retrieval interface — returns top-K chunks for a given source
# ---------------------------------------------------------------------------

class SourceRetriever:
    """
    Wraps one FAISS index + chunk_id map for a single source.
    Loaded lazily — only one source in RAM at a time during training
    (same memory-safe pattern as script_4).
    """
    def __init__(self, src: str, indices_dir: Path):
        self.src   = src
        p          = indices_dir / src
        self.index = faiss.read_index(str(p / "index.faiss"))
        self.cids  = np.load(p / "chunk_ids.npy", allow_pickle=True)
        print(f"  SourceRetriever[{src}]: {self.index.ntotal:,} vectors")

    def retrieve(self, query_emb: np.ndarray, k: int = TOP_K) -> list[str]:
        """
        Args:
            query_emb: (768,) float32 numpy array (single query).
        Returns:
            List of chunk_ids, length k.
        """
        q = query_emb.reshape(1, -1).astype(np.float32)
        _, ids = self.index.search(q, k)
        return [self.cids[i] for i in ids[0]]


# Load all retrievers (they fit in RAM together — ~10GB at float32 for 5 sources)
print("\nLoading FAISS indices...")
retrievers = {src: SourceRetriever(src, INDICES_DIR) for src in SOURCES}


# ---------------------------------------------------------------------------
# BGE query encoder (kept in RAM during training — it's small, 438MB)
# ---------------------------------------------------------------------------

print(f"\nLoading BGE encoder: {BGE_NAME}")
bge_tok   = AutoTokenizer.from_pretrained(BGE_NAME)
bge_model = AutoModel.from_pretrained(BGE_NAME, torch_dtype=torch.float32).to(DEVICE).eval()


@torch.inference_mode()
def encode_query_single(query: str) -> np.ndarray:
    """Returns (768,) float32 embedding for one query."""
    enc = bge_tok(QUERY_PREFIX + query, return_tensors="pt",
                  truncation=True, max_length=MAX_QUERY_LEN).to(DEVICE)
    emb = bge_model(**enc).last_hidden_state[:, 0]
    return F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# AnswerFormer — Llama 3.1 8B Instruct (4-bit) with LoRA
# ---------------------------------------------------------------------------

print(f"\nLoading AnswerFormer: {LLM_NAME}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

llm_tok = AutoTokenizer.from_pretrained(LLM_NAME)
llm_tok.pad_token = llm_tok.eos_token
llm_tok.padding_side = "right"

llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    quantization_config = bnb_config,
    device_map          = "auto",
    torch_dtype         = torch.bfloat16,
)

lora_config = LoraConfig(
    r             = LORA_R,
    lora_alpha    = LORA_ALPHA,
    lora_dropout  = LORA_DROPOUT,
    target_modules= LORA_TARGETS,
    task_type     = TaskType.CAUSAL_LM,
    bias          = "none",
)
llm = get_peft_model(llm, lora_config)
llm.print_trainable_parameters()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def oracle_label(item: dict) -> int | None:
    scores = item["dataset_score"]
    max_s  = max(scores.values())
    if max_s == 0:
        return None
    winners = [src for src in SOURCES if scores.get(src, 0) == max_s]
    return SOURCE_TO_IDX[winners[0]]


def load_data(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    records = []
    for item in data:
        label = oracle_label(item)
        if label is None:
            continue
        records.append({
            "query":  item["query"],
            "answer": item.get("answer", ""),
            "label":  label,
        })
    return records


class JointDataset(Dataset):
    def __init__(self, records: list[dict], emb_cache: Path, split: str):
        self.records = records
        cache_path   = emb_cache / f"joint_{split}_embs.npy"

        if cache_path.exists():
            self.embs = np.load(cache_path)
        else:
            print(f"  Encoding {split} queries for joint training...")
            embs = np.stack([encode_query_single(r["query"]) for r in tqdm(records)])
            np.save(cache_path, embs)
            self.embs = embs

        self.embs   = torch.from_numpy(self.embs).float()
        self.labels = torch.tensor([r["label"] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx], self.records[idx]


print("\nLoading datasets...")
DEV_FILE   = "mmrag_dev.json"
TRAIN_FILE = "mmrag_train.json"
if Path(TRAIN_FILE).exists():
    train_recs = load_data(TRAIN_FILE)
    val_recs   = load_data(DEV_FILE)
else:
    recs = load_data(DEV_FILE)
    random.seed(42)
    random.shuffle(recs)
    split      = int(0.8 * len(recs))
    train_recs = recs[:split]
    val_recs   = recs[split:]

train_ds = JointDataset(train_recs, EMB_CACHE, "train")
val_ds   = JointDataset(val_recs,   EMB_CACHE, "val")


def collate_fn(batch):
    embs, labels, records = zip(*batch)
    return torch.stack(embs), torch.stack(labels), list(records)


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)


# ---------------------------------------------------------------------------
# Build prompt and compute LM loss
# ---------------------------------------------------------------------------

def build_prompt(query: str, evidence_chunks: list[str]) -> str:
    evidence_text = "\n\n".join(
        f"[Passage {i+1}] {chunk}" for i, chunk in enumerate(evidence_chunks))
    return (
        f"<|system|>\nYou are a precise QA assistant. Answer the question "
        f"using only the provided passages.\n"
        f"<|user|>\nPassages:\n{evidence_text}\n\nQuestion: {query}\n"
        f"<|assistant|>\nAnswer:"
    )


def answer_loss(query: str, answer: str, evidence: list[str]) -> torch.Tensor:
    """
    Compute cross-entropy on answer tokens only.
    Uses teacher forcing — answer string is the target.
    """
    prompt    = build_prompt(query, evidence)
    full_text = prompt + " " + answer

    # Tokenize full sequence
    enc       = llm_tok(
        full_text, return_tensors="pt",
        max_length=MAX_CTX_LEN, truncation=True,
    ).to(DEVICE)

    input_ids = enc["input_ids"]

    # Find where the answer starts — compute loss only on answer tokens
    prompt_ids = llm_tok(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    labels        = input_ids.clone()
    labels[:, :prompt_len] = -100     # ignore prompt tokens in loss

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = llm(input_ids=input_ids, labels=labels)

    return out.loss


# ---------------------------------------------------------------------------
# Joint training loop
# ---------------------------------------------------------------------------

def get_tau(step: int, total_steps: int) -> float:
    """Linear temperature annealing: TAU_START → TAU_END."""
    frac = min(step / total_steps, 1.0)
    return TAU_START + frac * (TAU_END - TAU_START)


def run_joint_training(seed: int, sf_ckpt_path: Path | None = None):
    """Full joint training run for one seed."""
    print(f"\n{'='*60}")
    print(f"Joint Training  |  seed={seed}  |  α={ALPHA}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # SourceFormer
    sf_model = build_sourceformer(SF_VARIANT).to(DEVICE)
    if sf_ckpt_path and sf_ckpt_path.exists():
        ckpt_data = torch.load(sf_ckpt_path, map_location=DEVICE)
        sf_model.load_state_dict(ckpt_data["state_dict"])
        print(f"  Loaded pre-trained SourceFormer from {sf_ckpt_path} "
              f"(val_acc={ckpt_data['val_acc']:.4f})")
    else:
        print("  WARNING: No pre-trained SourceFormer found. "
              "Starting from random init (expect routing collapse risk).")

    # Separate optimizers — different LRs
    sf_optimizer   = torch.optim.AdamW(
        sf_model.parameters(), lr=LR_SF, weight_decay=WEIGHT_DECAY)
    lora_optimizer = torch.optim.AdamW(
        [p for p in llm.parameters() if p.requires_grad],
        lr=LR_LORA, weight_decay=WEIGHT_DECAY,
    )

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def warmup_cosine(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

    sf_scheduler   = torch.optim.lr_scheduler.LambdaLR(sf_optimizer,   warmup_cosine)
    lora_scheduler = torch.optim.lr_scheduler.LambdaLR(lora_optimizer, warmup_cosine)

    best_val_em  = 0.0
    best_ckpt    = CKPT_DIR / f"joint_seed{seed}_best.pt"
    patience_ctr = 0
    global_step  = 0
    log          = []

    sf_optimizer.zero_grad()
    lora_optimizer.zero_grad()

    for epoch in range(1, EPOCHS + 1):
        sf_model.train()
        llm.train()

        ep_loss_route = 0.0
        ep_loss_ans   = 0.0
        ep_route_acc  = 0.0
        n_batches     = 0

        for step, (embs, labels, records) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}")):

            tau       = get_tau(global_step, total_steps)
            embs      = embs.to(DEVICE)
            labels    = labels.to(DEVICE)

            # --- Routing ---
            logits  = sf_model(embs)           # (B, K)
            routing = gumbel_softmax(logits, tau=tau, hard=True)   # (B, K)

            # L_route — supervised cross-entropy
            L_route = F.cross_entropy(logits, labels)
            route_acc = (logits.argmax(-1) == labels).float().mean().item()

            # --- Answer loss (per item in batch) ---
            L_ans_total = torch.tensor(0.0, device=DEVICE, requires_grad=False)
            for i, record in enumerate(records):
                src_idx  = routing[i].argmax().item()           # hard selected source
                src_name = SOURCES[src_idx]

                # Retrieve from selected source
                q_emb    = embs[i].detach().cpu().numpy()
                cids     = retrievers[src_name].retrieve(q_emb, k=TOP_K)
                evidence = [chunk_store.get(src_name, cid) for cid in cids]

                # Answer loss
                L_ans_i  = answer_loss(record["query"], record["answer"], evidence)

                # Weight by Gumbel-Softmax soft weight for gradient flow
                # routing[i, src_idx] ≈ 1.0 (hard, but gradient flows through soft)
                L_ans_total = L_ans_total + routing[i, src_idx] * L_ans_i

            L_ans = L_ans_total / len(records)

            # --- Joint loss ---
            L_total = ALPHA * L_route + (1.0 - ALPHA) * L_ans

            # Normalize for gradient accumulation
            (L_total / GRAD_ACCUM).backward()

            ep_loss_route += L_route.item()
            ep_loss_ans   += L_ans.item()
            ep_route_acc  += route_acc
            n_batches     += 1
            global_step   += 1

            # Gradient accumulation step
            if global_step % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(sf_model.parameters(), GRAD_CLIP)
                nn.utils.clip_grad_norm_(
                    [p for p in llm.parameters() if p.requires_grad], GRAD_CLIP)
                sf_optimizer.step();   sf_scheduler.step()
                lora_optimizer.step(); lora_scheduler.step()
                sf_optimizer.zero_grad()
                lora_optimizer.zero_grad()

        # --- Validation ---
        sf_model.eval()
        llm.eval()
        val_em_scores = []
        val_route_acc = []

        with torch.no_grad():
            for embs, labels, records in tqdm(val_loader, desc="val", leave=False):
                embs   = embs.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = sf_model(embs)
                preds  = logits.argmax(-1)
                val_route_acc.extend((preds == labels).cpu().tolist())

                for i, record in enumerate(records):
                    src_idx  = preds[i].item()
                    src_name = SOURCES[src_idx]
                    q_emb    = embs[i].cpu().numpy()
                    cids     = retrievers[src_name].retrieve(q_emb, k=TOP_K)
                    evidence = [chunk_store.get(src_name, cid) for cid in cids]

                    # Greedy decode
                    prompt  = build_prompt(record["query"], evidence)
                    enc     = llm_tok(prompt, return_tensors="pt",
                                      max_length=MAX_CTX_LEN,
                                      truncation=True).to(DEVICE)
                    gen_ids = llm.generate(
                        **enc, max_new_tokens=MAX_GEN_LEN,
                        do_sample=False, pad_token_id=llm_tok.eos_token_id)
                    gen_text = llm_tok.decode(
                        gen_ids[0][enc["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip().lower()

                    gold_em  = record["answer"].strip().lower()
                    val_em_scores.append(float(gen_text == gold_em))

        val_em      = np.mean(val_em_scores)
        val_rac     = np.mean(val_route_acc)
        tr_l_route  = ep_loss_route / n_batches
        tr_l_ans    = ep_loss_ans   / n_batches
        tr_rac      = ep_route_acc  / n_batches

        log.append({
            "epoch":          epoch,
            "tau":            tau,
            "train_L_route":  tr_l_route,
            "train_L_ans":    tr_l_ans,
            "train_route_acc": tr_rac,
            "val_route_acc":  val_rac,
            "val_em":         val_em,
        })

        flag = ""
        if val_em > best_val_em:
            best_val_em = val_em
            torch.save({
                "epoch":          epoch,
                "seed":           seed,
                "sf_state_dict":  sf_model.state_dict(),
                "val_em":         val_em,
                "val_route_acc":  val_rac,
            }, best_ckpt)
            llm.save_pretrained(str(CKPT_DIR / f"joint_seed{seed}_lora"))
            patience_ctr = 0
            flag = "  ← best"
        else:
            patience_ctr += 1

        print(f"  Ep {epoch:3d}  τ={tau:.3f}  "
              f"Lr={tr_l_route:.4f}  La={tr_l_ans:.4f}  "
              f"tr_rac={tr_rac:.3f}  vl_rac={val_rac:.3f}  "
              f"val_EM={val_em:.4f}{flag}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"\n  Best val EM: {best_val_em:.4f}  → {best_ckpt}")
    return best_val_em, log


# ---------------------------------------------------------------------------
# Main — run joint training for all seeds
# ---------------------------------------------------------------------------

all_logs = {}
all_ems  = []

for seed in SEEDS:
    sf_ckpt = CKPT_DIR / f"sourceformer_{SF_VARIANT}_seed{seed}_best.pt"
    em, log = run_joint_training(seed, sf_ckpt_path=sf_ckpt)
    all_ems.append(em)
    all_logs[f"seed_{seed}"] = {"best_val_em": em, "epochs": log}

# Save logs
with open("joint_training_log.json", "w") as f:
    json.dump(all_logs, f, indent=2)

print(f"\n{'='*60}")
print("JOINT TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Val EM across seeds: {all_ems}")
print(f"Mean EM: {np.mean(all_ems):.4f} ± {np.std(all_ems):.4f}")
print(f"Logs saved to joint_training_log.json")
print(f"\nNext: Phase 6 (baselines) — run random, oracle, classifier, majority routing")
