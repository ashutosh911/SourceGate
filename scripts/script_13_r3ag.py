"""
script_13_r3ag.py — R³AG Baseline for SourceFormer (ROCm edition)

Implements R³AG (Zhao et al., 2026, arXiv:2604.22849) adapted to K=3 source
routing on the mmRAG benchmark.

R³AG decomposes retriever capability into two learnable dimensions:
  - Retrieval Quality  (φ_r): how well a source returns relevant evidence,
                              supervised by dataset_score binary relevance labels.
  - Generation Utility (φ_g): how well a source's evidence supports correct
                              answer generation, supervised by gold-answer NLL
                              on training queries (teacher-forcing, single LLM
                              forward pass per source per query).

Architecture:
  K=3 learnable source-type capability embeddings per dimension (e_r^k, e_g^k).
  Query projection heads (φ_r, φ_g) map BGE embeddings to capability space.
  Multi-head cross-attention fuses [e_r^k, e_g^k] under query guidance.
  Cosine similarity between projected query and fused representation → routing.

Training (two stages):
  Stage 0  Precompute generation utility labels via gold-answer NLL (LLM fp16,
           single forward pass, ROCm-safe). Cacheable / resumable.
  Stage 1  Train capability encoders (proj_r, proj_g, source_cap_r, source_cap_g)
           with InfoNCE contrastive loss on retrieval quality and generation utility
           labels independently.
  Stage 2  Freeze capability encoders; train multi-head attention fusion
           (proj_q_fuse, attn, proj_out) with CE routing loss + regularization
           that prevents the fused representation from drifting away from e_g^k.

Evaluation:
  Stage 2 routing logits → argmax → predicted source type.
  Reports routing accuracy, macro accuracy, per-type accuracy on dev and test.

USAGE:
  python script_13_r3ag.py --skip_phase0      # smoke-test: no LLM needed
  python script_13_r3ag.py --resume_phase0    # resume interrupted LLM pass
  python script_13_r3ag.py                    # full run (~4h Phase 0 + <5min train)

COMPUTE:
  Phase 0:   RX 9060 XT (fp16 Llama, rocm_exp env), ~3.5-4h for 3,072 train queries
  Stage 1/2: CPU or GPU, tiny model (<1M params), <5 min total
"""

import json, argparse, time, sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# =============================================================================
# Config
# =============================================================================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BGE_NAME      = "BAAI/bge-base-en-v1.5"
LLM_NAME      = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "

TRAIN_FILE  = "mmrag_train.json"
DEV_FILE    = "mmrag_dev.json"
TEST_FILE   = "mmrag_test.json"
INDICES_DIR = Path("faiss_indices")
CHUNK_DB    = Path("chunk_texts.db")
EMB_DIR     = Path("query_emb_cache")
RESULTS_DIR = Path("phase6_results")
# Suffix appended to every artifact this run writes. Empty by default, so
# the default (R3AG-full) invocation writes exactly the filenames it always
# did; set via --tag to keep a variant\'s outputs from clobbering them.
OUT_TAG = ""
RESULTS_DIR.mkdir(exist_ok=True)
EMB_DIR.mkdir(exist_ok=True)

SOURCE_TYPES      = ["text", "table", "kg"]
TYPE_TO_DATASETS  = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
SOURCE_IDX        = {t: i for i, t in enumerate(SOURCE_TYPES)}
# Retained for any caller that imports it; chunk-text lookup no longer uses
# it (see build_pos_cache). "kg" corrected from the old "g.%", which missed
# the 1,223,020 "m.%" Freebase ids.
DS_PREFIX         = {"nq": "nq_%", "triviaqa": "triviaqa_%",
                     "ott": "ott_%", "tat": "tat_%", "kg": "m.%"}

TOP_K         = 10
MAX_CTX_CHARS = 800
ENCODE_BATCH  = 64

# R³AG model hyperparameters
CAP_DIM        = 256    # capability embedding dimension
NUM_HEADS      = 4      # attention heads in fusion
TEMPERATURE    = 0.07   # InfoNCE temperature (standard CLIP-style)
LR_STAGE1      = 3e-4
LR_STAGE2      = 1e-4
EPOCHS_STAGE1  = 40
EPOCHS_STAGE2  = 30
BATCH_SIZE     = 64
LAMBDA_REG     = 0.1    # regularization weight stage 2
PATIENCE       = 10     # early stopping patience (both stages)


# =============================================================================
# Label helpers
# =============================================================================
def hard_label_k3(item):
    """Retrieval quality hard label: argmax of per-source dataset_score sum."""
    s = item["dataset_score"]
    return int(np.argmax([
        s.get("nq", 0) + s.get("triviaqa", 0),
        s.get("ott", 0) + s.get("tat", 0),
        s.get("kg", 0),
    ]))


# =============================================================================
# BGE encoding (with per-split cache)
# =============================================================================
def encode_split(records, split_name):
    cache = EMB_DIR / f"{split_name}_embs.npy"
    if cache.exists():
        embs = np.load(cache)
        if embs.shape[0] == len(records):
            print(f"  BGE cache hit [{split_name}]: shape={embs.shape}")
            return embs
        print(f"  Cache size mismatch ({embs.shape[0]} vs {len(records)}); re-encoding.")
    print(f"  Encoding {len(records)} {split_name} queries with BGE...")
    tok = AutoTokenizer.from_pretrained(BGE_NAME)
    bge = AutoModel.from_pretrained(BGE_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    queries = [QUERY_PREFIX + r["query"] for r in records]
    n   = len(queries)
    out = np.empty((n, 768), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, n, ENCODE_BATCH), desc=f"BGE {split_name}"):
            e   = min(s + ENCODE_BATCH, n)
            enc = tok(queries[s:e], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(DEVICE)
            emb = bge(**enc).last_hidden_state[:, 0]
            out[s:e] = F.normalize(emb.float(), p=2, dim=1).cpu().numpy()
    del bge, tok
    torch.cuda.empty_cache()
    np.save(cache, out)
    print(f"  Saved BGE cache: {cache}")
    return out


# =============================================================================
# FAISS + SQLite chunk lookup (same infrastructure as existing scripts)
# =============================================================================
def load_indices():
    idx = {}
    for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
        idx[ds] = faiss.read_index(str(INDICES_DIR / ds / "index.faiss"))
        print(f"  {ds}: {idx[ds].ntotal:,} vectors")
    return idx


_pos_cache: dict = {}


def build_pos_cache(ds_name: str):
    """Map FAISS position -> chunk id from the index's own chunk_ids.npy.

    Bug fix: this previously rebuilt the mapping with
    "SELECT id, text FROM chunks WHERE id LIKE ? ORDER BY id", which was
    wrong twice over. (1) DS_PREFIX["kg"] was "g.%", matching 4,094 of
    1,227,114 Freebase ids, so kg contexts came out essentially empty.
    (2) The FAISS indices were built in INSERTION order, not sorted by id,
    so 0% of positions lined up for ANY dataset -- get_chunk_texts returned
    the wrong chunk text even where the prefix was right. chunk_ids.npy is
    the authoritative position->id record written at index-build time.
    """
    if ds_name in _pos_cache:
        return
    ids = np.load(INDICES_DIR / ds_name / "chunk_ids.npy", allow_pickle=True)
    _pos_cache[ds_name] = list(ids)
    print(f"  {ds_name}: {len(ids):,} chunk ids in pos cache")


def get_chunk_texts(ds_name: str, positions: list) -> list:
    build_pos_cache(ds_name)
    ids = _pos_cache[ds_name]
    want = [ids[i] for i in positions if 0 <= i < len(ids)]
    if not want:
        return []
    conn = sqlite3.connect(str(CHUNK_DB))
    q = ",".join("?" * len(want))
    rows = dict(conn.execute(
        f"SELECT id, text FROM chunks WHERE id IN ({q})", want).fetchall())
    conn.close()
    return [rows[c] for c in want if c in rows]


def retrieve_context(q_emb: np.ndarray, source_type: str,
                     faiss_indices: dict, k: int = TOP_K) -> str:
    texts = []
    for ds in TYPE_TO_DATASETS[source_type]:
        _, ids = faiss_indices[ds].search(q_emb[None].astype(np.float32), k)
        texts += get_chunk_texts(ds, ids[0].tolist())
    return "\n\n".join(texts)[:MAX_CTX_CHARS]


# =============================================================================
# LLM loading (fp16, ROCm-safe — no bitsandbytes 4-bit)
# =============================================================================
def load_llm():
    print(f"\nLoading {LLM_NAME} → GPU (fp16, ROCm)...")
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "14GiB", "cpu": "4GiB"},
        low_cpu_mem_usage=True,
    ).eval()
    vram = torch.cuda.memory_allocated(0) / 1024 ** 3
    print(f"  LLM loaded. VRAM: {vram:.1f} GB")
    return tok, llm


def score_gen_utility(query: str, answer: str, context: str,
                      tok, llm) -> float:
    """
    Gold-answer NLL given context + question (teacher forcing).
    Implements the generation utility signal from R³AG §4.2.1:
    lower NLL = better generation support = higher utility.
    Returns negative NLL (higher = better).

    Note: uses gold answer tokens (not question tokens as in PrefRAG-Conf).
    This is the proper generation utility signal: log P(a* | context, q).
    """
    prefix = (
        "Answer the following question using only the provided context. "
        "Be concise.\n\nContext:\n" + context + "\n\nQuestion: " + query
        + "\n\nAnswer: "
    )
    full = prefix + str(answer)

    enc_prefix = tok(prefix, return_tensors="pt",
                     truncation=True, max_length=900)
    enc_full   = tok(full, return_tensors="pt",
                     truncation=True, max_length=1024).to(DEVICE)

    prefix_len = enc_prefix["input_ids"].shape[1]
    seq_len    = enc_full["input_ids"].shape[1]
    if prefix_len >= seq_len:
        return -1e9   # answer was truncated away

    labels = enc_full["input_ids"].clone()
    labels[:, :prefix_len] = -100   # mask prefix; NLL only on answer tokens

    with torch.no_grad():
        out = llm(**enc_full, labels=labels)

    return -out.loss.item()   # negate: higher = lower NLL = better utility


# =============================================================================
# Phase 0: Precompute generation utility labels on training set
# =============================================================================
def compute_gen_utility_labels(train_data: list, train_embs: np.ndarray,
                               faiss_indices: dict,
                               resume: bool = False) -> list:
    """
    For each training query, run a single LLM forward pass (fp16) per source type,
    scoring gold-answer NLL given the top-10 retrieved context.
    Returns list of dicts sorted by query_idx.

    Estimated time: ~3.5-4h on RX 9060 XT for 3,072 queries × 3 sources.
    """
    cache_path = RESULTS_DIR / "r3ag_gen_utility_train.json"

    results, done_ids = [], set()
    if resume and cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        done_ids = {r["query_idx"] for r in results}
        print(f"  Resuming Phase 0 from {len(done_ids)} completed queries.")

    if len(done_ids) == len(train_data):
        print(f"  Phase 0 complete ({len(results)} cached).")
        results.sort(key=lambda x: x["query_idx"])
        return results

    tok, llm = load_llm()
    t0 = time.time()

    for i, item in enumerate(tqdm(train_data, desc="Phase0 gen-utility")):
        if i in done_ids:
            continue

        query  = item["query"]
        answer = item.get("answer", "")   # gold answer string

        nll_scores = {}
        for src in SOURCE_TYPES:
            ctx = retrieve_context(train_embs[i], src, faiss_indices)
            try:
                nll_scores[src] = float(score_gen_utility(query, answer, ctx, tok, llm))
            except Exception as e:
                print(f"\n  WARNING query {i} src={src}: {e} — assigning -1e9")
                nll_scores[src] = -1e9

        best_src = max(nll_scores, key=nll_scores.get)
        results.append({
            "query_idx":  i,
            "nll_scores": nll_scores,
            "best_source": best_src,
            "best_label":  SOURCE_IDX[best_src],
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_done  = len(results)
            eta     = elapsed / n_done * (len(train_data) - n_done)
            print(f"\n  [{i+1}/{len(train_data)}] "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
            with open(cache_path, "w") as f:
                json.dump(results, f)

    del llm, tok
    torch.cuda.empty_cache()
    results.sort(key=lambda x: x["query_idx"])
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Phase 0 done. Saved: {cache_path}")
    return results


# =============================================================================
# R³AG Model
# =============================================================================
class R3AGModel(nn.Module):
    """
    R³AG adapted to K=3 source-type routing on mmRAG.

    Source-type capability embeddings play the role of "retriever representations"
    from §4.1 of the original paper, now encoding text/table/KG source types
    instead of different retrieval algorithms (BM25, DPR, etc.).

    Architecture (§4.1):
      source_cap_r  K learnable retrieval-quality capability embeddings e_r^k
      source_cap_g  K learnable generation-utility capability embeddings e_g^k
      proj_r        query → retrieval-quality feature space (φ_r)
      proj_g        query → generation-utility feature space (φ_g)
      proj_q_fuse   query projection for Stage 2 cross-attention
      attn          multi-head cross-attention fusing [e_r^k, e_g^k] under q guidance
      proj_out      post-attention projection

    Stage 1 routing (contrastive heads only):
      logits = (φ_r(q) · e_r^T + φ_g(q) · e_g^T) / 2   (sum of contrastive heads)

    Stage 2 routing (full model):
      For each source k: fused_k = attn(proj_q(q), [e_r^k, e_g^k])
      score_k = proj_q(q) · fused_k  (cosine, unnormalized)
    """

    def __init__(self, query_dim: int = 768, cap_dim: int = CAP_DIM,
                 K: int = 3, num_heads: int = NUM_HEADS):
        super().__init__()
        self.K       = K
        self.cap_dim = cap_dim

        # Learnable source-type capability embeddings
        self.source_cap_r = nn.Parameter(torch.randn(K, cap_dim) * 0.02)
        self.source_cap_g = nn.Parameter(torch.randn(K, cap_dim) * 0.02)

        # Stage 1 query projection heads
        self.proj_r = nn.Sequential(
            nn.Linear(query_dim, cap_dim),
            nn.GELU(),
            nn.LayerNorm(cap_dim),
        )
        self.proj_g = nn.Sequential(
            nn.Linear(query_dim, cap_dim),
            nn.GELU(),
            nn.LayerNorm(cap_dim),
        )

        # Stage 2 cross-attention fusion
        self.proj_q_fuse = nn.Linear(query_dim, cap_dim)
        self.attn        = nn.MultiheadAttention(
            cap_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.proj_out = nn.Linear(cap_dim, cap_dim)
        self.ln_out   = nn.LayerNorm(cap_dim)

    def stage1_logits(self, q: torch.Tensor, temp: float = TEMPERATURE):
        """
        InfoNCE logits for Stage 1 contrastive training.
        Returns (logits_r, logits_g), each shape (B, K).
        """
        q_r    = F.normalize(self.proj_r(q), dim=-1)       # (B, d)
        q_g    = F.normalize(self.proj_g(q), dim=-1)       # (B, d)
        e_r    = F.normalize(self.source_cap_r, dim=-1)    # (K, d)
        e_g    = F.normalize(self.source_cap_g, dim=-1)    # (K, d)
        logits_r = (q_r @ e_r.T) / temp                    # (B, K)
        logits_g = (q_g @ e_g.T) / temp                    # (B, K)
        return logits_r, logits_g

    def stage2_logits(self, q: torch.Tensor) -> torch.Tensor:
        """
        Stage 2 routing via capability fusion (§4.1 Capability Fusion).
        For each source k, cross-attend q over [e_r^k, e_g^k]; route by cosine.

        Uses a single batched attention call (B*K queries) for efficiency.
        Returns routing logits of shape (B, K).
        """
        B = q.shape[0]
        q_proj = self.proj_q_fuse(q)                                    # (B, d)

        # Build batched key-value: each source provides 2 capability vectors
        # Shape: (B*K, 2, d)
        e_r_exp = self.source_cap_r.unsqueeze(0).expand(B, -1, -1)      # (B, K, d)
        e_g_exp = self.source_cap_g.unsqueeze(0).expand(B, -1, -1)      # (B, K, d)
        kv = torch.stack([e_r_exp, e_g_exp], dim=2)                      # (B, K, 2, d)
        kv = kv.reshape(B * self.K, 2, self.cap_dim)                     # (B*K, 2, d)

        # Query: replicate q_proj for each source
        q_exp = q_proj.unsqueeze(1).expand(-1, self.K, -1)               # (B, K, d)
        q_exp = q_exp.reshape(B * self.K, 1, self.cap_dim)               # (B*K, 1, d)

        fused, _ = self.attn(q_exp, kv, kv)                              # (B*K, 1, d)
        fused    = fused.squeeze(1)                                       # (B*K, d)
        fused    = self.ln_out(self.proj_out(fused))                      # (B*K, d)
        fused    = fused.reshape(B, self.K, self.cap_dim)                 # (B, K, d)

        # Routing score: dot product between projected query and fused capability
        logits = torch.einsum("bd,bkd->bk", q_proj, fused)               # (B, K)
        return logits

    def regularization_loss(self, q: torch.Tensor) -> torch.Tensor:
        """
        Stage 2 regularization (§4.2.2): prevent fused representation from
        deviating excessively from the generation utility embedding e_g^k.
        R_reg = mean over k of (1 - cos(fused_k, e_g^k)).
        """
        B = q.shape[0]
        q_proj = self.proj_q_fuse(q)

        e_r_exp = self.source_cap_r.unsqueeze(0).expand(B, -1, -1)
        e_g_exp = self.source_cap_g.unsqueeze(0).expand(B, -1, -1)
        kv = torch.stack([e_r_exp, e_g_exp], dim=2).reshape(B * self.K, 2, self.cap_dim)
        q_exp = q_proj.unsqueeze(1).expand(-1, self.K, -1).reshape(B * self.K, 1, self.cap_dim)

        fused, _ = self.attn(q_exp, kv, kv)
        fused    = fused.squeeze(1).reshape(B, self.K, self.cap_dim)      # (B, K, d)

        e_g_norm  = F.normalize(self.source_cap_g, dim=-1)                # (K, d)
        fused_norm = F.normalize(fused, dim=-1)                            # (B, K, d)
        cos_sim    = torch.einsum("bkd,kd->bk", fused_norm, e_g_norm)    # (B, K)
        return (1.0 - cos_sim).mean()


# =============================================================================
# Stage 1 training
# =============================================================================
def train_stage1(model: R3AGModel, train_embs: np.ndarray,
                 y_r: list, y_g: list, device: str) -> None:
    """
    Contrastive InfoNCE training of capability encoders.
    Trains: proj_r, proj_g, source_cap_r, source_cap_g.
    Other parameters (fusion, out) are not updated this stage.
    """
    model.train()
    stage1_params = (
        list(model.proj_r.parameters())
        + list(model.proj_g.parameters())
        + [model.source_cap_r, model.source_cap_g]
    )
    opt = torch.optim.AdamW(stage1_params, lr=LR_STAGE1, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STAGE1)

    embs_t = torch.from_numpy(train_embs).float()
    yr_t   = torch.tensor(y_r, dtype=torch.long)
    yg_t   = torch.tensor(y_g, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(embs_t, yr_t, yg_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_loss = float("inf")
    patience_count = 0
    ckpt_path = RESULTS_DIR / f"r3ag_stage1_best{OUT_TAG}.pt"

    for epoch in range(EPOCHS_STAGE1):
        epoch_loss = 0.0
        model.train()
        for q, labels_r, labels_g in loader:
            q, labels_r, labels_g = (
                q.to(device), labels_r.to(device), labels_g.to(device)
            )
            logits_r, logits_g = model.stage1_logits(q)
            loss = (
                F.cross_entropy(logits_r, labels_r)
                + F.cross_entropy(logits_g, labels_g)
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(stage1_params, 1.0)
            opt.step()
            epoch_loss += loss.item() * len(q)

        avg_loss = epoch_loss / len(dataset)
        sched.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Stage1 [{epoch+1:3d}/{EPOCHS_STAGE1}]  loss={avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Stage1 early stop at epoch {epoch+1}  best_loss={best_loss:.4f}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Stage1 done.  Best loss={best_loss:.4f}  Checkpoint: {ckpt_path}")


# =============================================================================
# Stage 2 training
# =============================================================================
def train_stage2(model: R3AGModel, train_embs: np.ndarray,
                 y_combined: list, device: str) -> None:
    """
    Multi-head attention fusion training (§4.2.2).
    Freezes capability encoders; trains: proj_q_fuse, attn, proj_out, ln_out.
    Loss = CE(routing_logits, y_combined) + λ * R_reg.
    Target y_combined: generation utility label (primary signal per R³AG §4.2.2).
    """
    # Freeze capability encoders
    for p in (
        list(model.proj_r.parameters())
        + list(model.proj_g.parameters())
        + [model.source_cap_r, model.source_cap_g]
    ):
        p.requires_grad_(False)

    stage2_params = (
        list(model.proj_q_fuse.parameters())
        + list(model.attn.parameters())
        + list(model.proj_out.parameters())
        + list(model.ln_out.parameters())
    )
    opt   = torch.optim.AdamW(stage2_params, lr=LR_STAGE2, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STAGE2)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_embs).float(),
        torch.tensor(y_combined, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_loss = float("inf")
    patience_count = 0
    ckpt_path = RESULTS_DIR / f"r3ag_stage2_best{OUT_TAG}.pt"

    for epoch in range(EPOCHS_STAGE2):
        epoch_loss = 0.0
        model.train()
        for q, labels in loader:
            q, labels = q.to(device), labels.to(device)
            logits = model.stage2_logits(q)
            reg    = model.regularization_loss(q)
            loss   = F.cross_entropy(logits, labels) + LAMBDA_REG * reg
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(stage2_params, 1.0)
            opt.step()
            epoch_loss += loss.item() * len(q)

        avg_loss = epoch_loss / len(dataset)
        sched.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Stage2 [{epoch+1:3d}/{EPOCHS_STAGE2}]  loss={avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Stage2 early stop at epoch {epoch+1}  best_loss={best_loss:.4f}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Stage2 done.  Best loss={best_loss:.4f}  Checkpoint: {ckpt_path}")

    # Unfreeze for any downstream use
    for p in model.parameters():
        p.requires_grad_(True)


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(model: R3AGModel, embs: np.ndarray,
             true_labels: list, split_name: str, device: str):
    """
    Run Stage 2 routing on embs; compare to true_labels (retrieval quality argmax).
    Returns (acc, macro, per_type_dict, preds_array).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(embs), BATCH_SIZE):
            q = torch.from_numpy(embs[s: s + BATCH_SIZE]).float().to(device)
            logits = model.stage2_logits(q)
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    preds  = np.array(preds)
    labels = np.array(true_labels)
    acc    = float((preds == labels).mean())

    per_type: dict = {}
    macro_parts: list = []
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (labels == idx)
        if mask.sum():
            pt = float((preds[mask] == idx).mean())
            per_type[t] = pt
            macro_parts.append(pt)
        else:
            per_type[t] = float("nan")

    macro = float(np.mean(macro_parts)) if macro_parts else float("nan")
    print(f"\n  [{split_name}] acc={acc:.4f}  macro={macro:.4f}  "
          f"text={per_type.get('text', float('nan')):.3f}  "
          f"table={per_type.get('table', float('nan')):.3f}  "
          f"kg={per_type.get('kg', float('nan')):.3f}")
    return acc, macro, per_type, preds


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_phase0", action="store_true",
                        help="Skip LLM generation utility computation; use "
                             "retrieval quality labels for both signals (smoke-test, "
                             "collapses Stage 1 dual signal to single signal).")
    parser.add_argument("--resume_phase0", action="store_true",
                        help="Resume interrupted Phase 0 from cache.")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Limit training set size (debugging).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Random seeds for Stage 1/2 training.")
    parser.add_argument("--tag", default="",
                        help="Suffix for output filenames, e.g. '_rq'. Empty "
                             "(default) reproduces the original filenames. Use "
                             "a tag for any variant run so it does not "
                             "overwrite the R3AG-full artifacts.")
    args = parser.parse_args()
    global OUT_TAG
    OUT_TAG = args.tag

    print("=" * 70)
    print("R³AG  (Zhao et al., 2026 — arXiv:2604.22849)")
    print("Adaptation: K=3 source-type routing on mmRAG benchmark")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with open(TRAIN_FILE) as f: train_data = json.load(f)
    with open(DEV_FILE)   as f: dev_data   = json.load(f)
    with open(TEST_FILE)  as f: test_data  = json.load(f)
    if args.n_train:
        train_data = train_data[: args.n_train]
    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # ------------------------------------------------------------------
    # BGE embeddings (cached per split)
    # ------------------------------------------------------------------
    print("\nBGE embeddings:")
    train_embs = encode_split(train_data, "train")
    dev_embs   = encode_split(dev_data,   "dev")
    test_embs  = encode_split(test_data,  "test")

    # ------------------------------------------------------------------
    # Retrieval quality labels (y_r) — from dataset_score, free
    # ------------------------------------------------------------------
    y_r_train = [hard_label_k3(x) for x in train_data]
    y_r_dev   = [hard_label_k3(x) for x in dev_data]
    y_r_test  = [hard_label_k3(x) for x in test_data]

    # ------------------------------------------------------------------
    # Generation utility labels (y_g) — Phase 0
    # ------------------------------------------------------------------
    if args.skip_phase0:
        print("\n[--skip_phase0] Using retrieval quality labels for both signals.")
        print("  NOTE: This disables R³AG's dual-signal contrastive training.")
        print("  Run without --skip_phase0 for the proper R³AG baseline.")
        y_g_train = y_r_train.copy()
    else:
        print("\nPhase 0: Computing generation utility labels (gold-answer NLL)...")
        print("Loading FAISS indices and chunk position cache...")
        faiss_indices = load_indices()
        for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
            build_pos_cache(ds)

        gen_utility = compute_gen_utility_labels(
            train_data, train_embs, faiss_indices,
            resume=args.resume_phase0,
        )
        # Align to train_data order (results are sorted by query_idx after Phase 0)
        y_g_train = [gen_utility[i]["best_label"] for i in range(len(train_data))]

        # Free FAISS and chunk cache before training (frees RAM)
        del faiss_indices
        _pos_cache.clear()
        torch.cuda.empty_cache()

        # Label agreement stats (informative)
        agree = sum(r == g for r, g in zip(y_r_train, y_g_train))
        print(f"\n  Retrieval quality / generation utility agreement: "
              f"{agree}/{len(y_r_train)} = {agree/len(y_r_train):.1%}")
        print("  (Disagreements motivate R³AG's dual-signal approach.)")

    # Stage 2 target: generation utility is the primary signal per R³AG §4.2.2
    y_combined_train = y_g_train

    # ------------------------------------------------------------------
    # Multi-seed training
    # ------------------------------------------------------------------
    all_test_macros = []
    all_test_per_type = []

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = R3AGModel(query_dim=768, cap_dim=CAP_DIM, K=3).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nR³AG parameters: {n_params:,}")

        # Stage 1: contrastive capability encoder training
        print(f"\nStage 1: Contrastive capability encoder training "
              f"(up to {EPOCHS_STAGE1} epochs)...")
        train_stage1(model, train_embs[: len(y_r_train)],
                     y_r_train, y_g_train, DEVICE)

        # Stage 1 interim evaluation (sum of both contrastive heads)
        model.eval()
        s1_preds = []
        with torch.no_grad():
            for s in range(0, len(test_embs), BATCH_SIZE):
                q = torch.from_numpy(test_embs[s: s + BATCH_SIZE]).float().to(DEVICE)
                lr, lg = model.stage1_logits(q)
                s1_preds.extend((lr + lg).argmax(dim=-1).cpu().numpy().tolist())
        s1_preds  = np.array(s1_preds)
        s1_labels = np.array(y_r_test)
        s1_acc    = float((s1_preds == s1_labels).mean())
        s1_macro  = float(np.mean([
            float((s1_preds[s1_labels == k] == k).mean())
            for k in range(3) if (s1_labels == k).sum() > 0
        ]))
        print(f"  Stage1 test (interpolated heads):  "
              f"acc={s1_acc:.4f}  macro={s1_macro:.4f}")

        # Stage 2: fusion training
        print(f"\nStage 2: Capability fusion training "
              f"(up to {EPOCHS_STAGE2} epochs)...")
        train_stage2(model, train_embs[: len(y_combined_train)],
                     y_combined_train, DEVICE)

        # Final evaluation
        print("\nEvaluation:")
        _, dev_macro, _, _ = evaluate(
            model, dev_embs, y_r_dev, f"Dev  (seed={seed})", DEVICE
        )
        test_acc, test_macro, test_pt, test_preds = evaluate(
            model, test_embs, y_r_test, f"Test (seed={seed})", DEVICE
        )

        all_test_macros.append(test_macro)
        all_test_per_type.append(test_pt)

        # Save per-seed predictions
        np.save(
            RESULTS_DIR / f"r3ag_test_preds{OUT_TAG}_seed{seed}.npy",
            test_preds.astype(np.int32),
        )
        torch.save(model.state_dict(),
                   RESULTS_DIR / f"r3ag_model{OUT_TAG}_seed{seed}.pt")

    # ------------------------------------------------------------------
    # Aggregate results across seeds
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"R³AG AGGREGATED RESULTS  (K=3, test set, n={len(test_data)})")
    print(f"{'='*70}")
    macro_arr = np.array(all_test_macros)
    print(f"  Routing macro: {macro_arr.mean():.4f} ± {macro_arr.std():.4f}  "
          f"(n={len(args.seeds)} seeds)")

    for t in SOURCE_TYPES:
        vals = [d[t] for d in all_test_per_type if not np.isnan(d.get(t, float("nan")))]
        if vals:
            print(f"  {t:6s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    print(f"\n  Comparison (K=3 test, best macro per method):")
    print(f"    Random:           macro=0.337")
    print(f"    BGE-confidence:   macro=0.661")
    print(f"    PrefRAG-Conf:     macro=0.334  (collapses to KG)")
    print(f"    MLP-HardCE (n=3): macro=0.702")
    print(f"    SF Phase 3 (n=3): macro=0.737")
    print(f"    R³AG (this):      macro={macro_arr.mean():.3f} ± {macro_arr.std():.3f}")

    # Save summary
    summary = {
        "method": "R3AG",
        "description": "R³AG capability-decomposed routing (Zhao et al., 2026, arXiv:2604.22849). "
                       "Adapted to K=3 source-type routing on mmRAG.",
        "n_train": len(train_data),
        "n_test":  len(test_data),
        "skip_phase0": args.skip_phase0,
        "seeds": args.seeds,
        "per_seed_macros": all_test_macros,
        "mean_macro": float(macro_arr.mean()),
        "std_macro":  float(macro_arr.std()),
        "per_seed_per_type": all_test_per_type,
        "hparams": {
            "cap_dim": CAP_DIM, "num_heads": NUM_HEADS,
            "temperature": TEMPERATURE,
            "lr_stage1": LR_STAGE1, "lr_stage2": LR_STAGE2,
            "epochs_stage1": EPOCHS_STAGE1, "epochs_stage2": EPOCHS_STAGE2,
            "lambda_reg": LAMBDA_REG, "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
        },
        "comparison": {
            "Random":          {"macro": 0.337},
            "BGE-confidence":  {"macro": 0.661},
            "PrefRAG-Conf":    {"macro": 0.334},
            "MLP-HardCE":      {"macro": 0.702},
            "SF_Phase3":       {"macro": 0.737},
        },
    }
    with open(RESULTS_DIR / f"r3ag_summary{OUT_TAG}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
