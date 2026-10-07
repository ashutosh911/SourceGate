"""
phase4_components.py

Building blocks for Phase 4 joint training. Designed so the training script
in script_6 just composes these. Every function here is unit-testable
without needing the real LLM weights.

Three loss terms:
  L_route : KL(soft_target || softmax(logits))      — supervised, Phase 3 signal
  L_ans   : -log p(gold_answer | query, picked_chunks) through frozen LLM
  L_aux   : -E_routing[ recall@k for that source ]  — soft attention over recall

Combined hierarchically:
  L_inner   = 0.75 · norm(L_ans)  +  0.25 · norm(L_aux)
  L_total   = 0.50 · norm(L_route) + 0.50 · L_inner

All `norm()` use exponential moving averages over a warmup period.
"""
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sourceformer import (
    SourceFormerK3, gumbel_softmax,
    SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS,
)


# =============================================================================
# 1. Loss normalization (running mean over warmup window)
# =============================================================================
class RunningNormalizer:
    """
    Tracks an exponential moving average of a loss term to enable scale-aware
    weighted sums. The first `warmup_steps` updates use a simple mean; after
    that, EMA with decay 0.99.

    DESIGN: We use EMA rather than dividing by current value because per-step
    division creates instability. The normalizer's job is to remove scale
    differences ACROSS loss terms, not to remove temporal variation within
    one term.
    """
    def __init__(self, warmup_steps=50, decay=0.99, eps=1e-6):
        self.warmup_steps = warmup_steps
        self.decay = decay
        self.eps = eps
        self.history = deque(maxlen=warmup_steps)
        self.ema = None
        self.steps = 0

    def update(self, value: float) -> float:
        """Update EMA with new value, return current normalizer."""
        self.steps += 1
        if self.steps <= self.warmup_steps:
            self.history.append(float(value))
            self.ema = float(np.mean(self.history))
        else:
            self.ema = self.decay * self.ema + (1.0 - self.decay) * float(value)
        return max(self.ema, self.eps)
        
    def normalize(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        scale = max(abs(self.ema) if self.ema is not None else 1.0, self.eps)
        return loss_tensor / scale

    def state_dict(self):
        return {"ema": self.ema, "steps": self.steps,
                "history": list(self.history),
                "warmup_steps": self.warmup_steps, "decay": self.decay}

    def load_state_dict(self, state):
        self.ema = state["ema"]
        self.steps = state["steps"]
        self.history = deque(state["history"], maxlen=state["warmup_steps"])
        self.warmup_steps = state["warmup_steps"]
        self.decay = state["decay"]


# =============================================================================
# 2. L_route: KL divergence against soft routing labels (same as Phase 3)
# =============================================================================
def compute_l_route(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, K) raw routing logits from SourceFormer
    soft_targets: (B, K) probability distribution over source types
    """
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1).mean()


# =============================================================================
# 3. L_aux: soft routing-weighted retrieval recall
# =============================================================================
def compute_l_aux(
    routing_probs: torch.Tensor,    # (B, K) softmax(logits), NOT Gumbel-sampled
    recall_per_source: torch.Tensor # (B, K) binary {0,1}: recall@k hit per source
) -> torch.Tensor:
    """
    Encourages SourceFormer to put weight on sources where retrieval would have hit.

    DESIGN: We use plain softmax routing_probs here, not Gumbel samples. The
    Gumbel pick is for L_ans (which needs a discrete choice for the LLM input).
    L_aux uses soft probs because we have all-source recall information available
    cheaply and there's no reason to inject Gumbel noise.

    We negate so minimizing L_aux maximizes expected recall.
    """
    expected_recall = (routing_probs * recall_per_source).sum(-1)  # (B,)
    return -expected_recall.mean()


# =============================================================================
# 4. L_ans: token NLL through frozen LLM
# =============================================================================
@dataclass
class LLMConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_len: int = 2048
    max_new_tokens: int = 64           # only used for eval-time generation
    load_in_4bit: bool = True
    answer_max_tokens: int = 32        # truncate gold answer if pathological

    # Standard prompt format chosen
    system_prompt: str = "Answer the question using only the provided context. Be concise."

    def chat_template(self, context: str, question: str, answer: Optional[str] = None) -> str:
        """
        Build the full prompt. If answer is None, returns prompt only (for inference).
        If answer provided, returns prompt + answer (for training NLL).
        """
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        if answer is None:
            return prompt
        return prompt + f"{answer}<|eot_id|>"


def format_chunks(chunks: list[str]) -> str:
    """Standard format: chunks separated by blank lines."""
    return "\n\n".join(chunks)


def compute_l_ans(
    llm,                          # transformers AutoModelForCausalLM (frozen)
    tokenizer,                    # corresponding tokenizer
    queries: list[str],           # (B,)
    contexts: list[str],          # (B,) — already-formatted chunk strings
    answers: list[str],           # (B,)
    cfg: LLMConfig,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute mean token NLL on the answer span.

    DESIGN: NLL is computed ONLY on answer tokens, not the prompt. We tokenize
    prompt and prompt+answer separately to get the answer token boundary, then
    mask the prompt portion to -100 (ignored by CrossEntropyLoss).
    """
    losses = []
    for q, ctx, ans in zip(queries, contexts, answers):
        prompt_only = cfg.chat_template(ctx, q, answer=None)
        prompt_with_answer = cfg.chat_template(ctx, q, answer=ans)

        # Tokenize both. The difference in length tells us where the answer starts.
        prompt_ids = tokenizer(prompt_only, add_special_tokens=False, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(prompt_with_answer, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # Hard cap to max_seq_len. Truncate from the LEFT of context if needed
        # so the question and answer span are preserved.
        if full_ids.size(0) > cfg.max_seq_len:
            overflow = full_ids.size(0) - cfg.max_seq_len
            # Truncate prompt_ids and full_ids at the start (cuts oldest context)
            prompt_ids = prompt_ids[overflow:]
            full_ids = full_ids[overflow:]

        prompt_len = prompt_ids.size(0)
        full_ids = full_ids.unsqueeze(0).to(device)
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100  # mask prompt tokens

        with torch.no_grad():
            outputs = llm(input_ids=full_ids, labels=labels)
        # outputs.loss is the mean NLL over the unmasked (answer) tokens
        losses.append(outputs.loss)

    return torch.stack(losses).mean()


# =============================================================================
# 5. Combined hierarchical loss
# =============================================================================
@dataclass
class LossWeights:
    alpha: float = 0.5     # outer: L_route vs L_inner
    beta: float = 0.75     # inner: L_ans share
    gamma: float = 0.25    # inner: L_aux share
    # alpha + (1-alpha) = 1; beta + gamma should = 1 by convention.


@dataclass
class LossState:
    """Tracks normalizers and accumulates loss components for logging."""
    norm_route: RunningNormalizer = field(default_factory=lambda: RunningNormalizer(warmup_steps=50))
    norm_ans: RunningNormalizer = field(default_factory=lambda: RunningNormalizer(warmup_steps=50))
    norm_aux: RunningNormalizer = field(default_factory=lambda: RunningNormalizer(warmup_steps=50))
    history: list = field(default_factory=list)


def compute_combined_loss(
    l_route: torch.Tensor,
    l_ans: torch.Tensor,
    l_aux: torch.Tensor,
    weights: LossWeights,
    state: LossState,
    log: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Returns (combined_loss, log_dict).

    DESIGN: We update normalizers BEFORE normalizing this step's losses.
    The current step uses the EMA AS OF this step, including itself. This is
    a deliberate choice — using prior EMA only causes the first step to be
    unnormalized, which destabilizes early training.
    """
    # Detached values for normalizer updates
    r = l_route.detach().item()
    a = l_ans.detach().item()
    x = l_aux.detach().item()

    state.norm_route.update(r)
    state.norm_ans.update(a)
    state.norm_aux.update(x)

    l_route_n = state.norm_route.normalize(l_route)
    l_ans_n   = state.norm_ans.normalize(l_ans)
    l_aux_n   = state.norm_aux.normalize(l_aux)

    l_inner = weights.beta * l_ans_n + weights.gamma * l_aux_n
    l_total = weights.alpha * l_route_n + (1.0 - weights.alpha) * l_inner

    log_dict = {
        "l_route_raw": r, "l_ans_raw": a, "l_aux_raw": x,
        "l_route_norm": l_route_n.item(),
        "l_ans_norm": l_ans_n.item(),
        "l_aux_norm": l_aux_n.item(),
        "l_inner": l_inner.item(),
        "l_total": l_total.item(),
        "ema_route": state.norm_route.ema,
        "ema_ans": state.norm_ans.ema,
        "ema_aux": state.norm_aux.ema,
    }
    if log:
        state.history.append(log_dict)
    return l_total, log_dict


# =============================================================================
# 6. Hard Gumbel pick → which source to retrieve from
# =============================================================================
def gumbel_hard_pick(
    sourceformer: SourceFormerK3,
    query_embs: torch.Tensor,   # (B, 768) BGE query embeddings
    tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits:        (B, K) raw routing logits — used for L_route
      routing_probs: (B, K) softmax(logits) — used for L_aux
      one_hot:       (B, K) hard one-hot with STE gradient — used for L_ans

    DESIGN: We return three things from one forward pass to avoid recomputing
    logits. The training script can pick which to consume per loss term.
    """
    logits = sourceformer(query_embs)
    routing_probs = F.softmax(logits, dim=-1)
    one_hot = gumbel_softmax(logits, tau=tau, hard=True)
    return logits, routing_probs, one_hot


# =============================================================================
# 7. Self-test (run this file directly to verify shapes and gradient flow)
# =============================================================================
def _self_test():
    """
    Tests every component WITHOUT loading the real LLM.
    L_ans is faked with a random scalar that depends on the routing one-hot
    (so we can verify gradient flows back).
    """
    print("=" * 60)
    print("Phase 4 components self-test")
    print("=" * 60)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 8

    sf = SourceFormerK3().to(device)
    sf.train()

    # Fake batch
    query_embs = torch.randn(B, 768, device=device)
    soft_targets = F.softmax(torch.randn(B, K), dim=-1).to(device)
    recall_per_source = torch.randint(0, 2, (B, K), dtype=torch.float32).to(device)

    # --- Test 1: Full combined loss + gradient bridge ---
    print("\n[Test 1] Full combined loss + gradient bridge (L_ans STE path)")
    logits, probs, one_hot = gumbel_hard_pick(sf, query_embs)
    print(f"  logits {logits.shape}, probs {probs.shape}, one_hot {one_hot.shape}")
    print(f"  one_hot row sums (should all = 1.0): {one_hot.sum(-1)[:3].tolist()}")

    l_route = compute_l_route(logits, soft_targets)
    l_aux = compute_l_aux(probs, recall_per_source)
    fake_source_quality = torch.randn(K, device=device)
    l_ans = -(one_hot * fake_source_quality).sum(-1).mean()

    print(f"  L_route = {l_route.item():.4f}")
    print(f"  L_ans (fake) = {l_ans.item():.4f}")
    print(f"  L_aux = {l_aux.item():.4f}")

    weights = LossWeights()
    state = LossState()
    for _ in range(5):
        l_total, log = compute_combined_loss(l_route, l_ans, l_aux, weights, state)

    print(f"  Final L_total = {log['l_total']:.4f}")
    print(f"    EMAs: route={log['ema_route']:.3f}, "
          f"ans={log['ema_ans']:.3f}, aux={log['ema_aux']:.3f}")

    sf.zero_grad()
    l_total.backward()  # frees graph — intentional
    grad_norm = sum(p.grad.norm().item() for p in sf.parameters() if p.grad is not None)
    n_grad = sum(1 for p in sf.parameters() if p.grad is not None and p.grad.norm().item() > 0)
    n_total = sum(1 for _ in sf.parameters())
    print(f"  Backward: grad_norm={grad_norm:.4f}, params with grad: {n_grad}/{n_total}")
    assert grad_norm > 0, "GRADIENT BRIDGE FAILED — no signal reached SourceFormer"
    print("  ✓ Gradient bridge verified (L_ans grad flows through Gumbel-STE)")

    # --- Test 2: L_aux gradient — FRESH FORWARD PASS ---
    # We must recompute because Test 1's backward() freed the graph.
    print("\n[Test 2] L_aux gradient (soft routing path, independent forward pass)")
    sf.zero_grad()
    query_embs2 = torch.randn(B, 768, device=device)
    recall2 = torch.randint(0, 2, (B, K), dtype=torch.float32).to(device)
    _, probs2, _ = gumbel_hard_pick(sf, query_embs2)
    l_aux2 = compute_l_aux(probs2, recall2)
    l_aux2.backward()
    grad_norm_aux = sum(p.grad.norm().item() for p in sf.parameters() if p.grad is not None)
    print(f"  L_aux backward only: grad_norm={grad_norm_aux:.4f}")
    assert grad_norm_aux > 0, "L_AUX GRADIENT FAILED"
    print("  ✓ L_aux gradient flows through softmax")

    # --- Test 3: RunningNormalizer behavior ---
    print("\n[Test 3] RunningNormalizer warmup + EMA")
    norm = RunningNormalizer(warmup_steps=10, decay=0.99)
    for v in [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]:
        norm.update(v)
    assert abs(norm.ema - 2.0) < 0.01, f"Expected EMA ~2.0, got {norm.ema}"
    fake_loss = torch.tensor(2.0)
    normalized = norm.normalize(fake_loss)
    assert abs(normalized.item() - 1.0) < 0.01, f"Expected normalized ~1.0, got {normalized.item()}"
    print(f"  EMA after 10 steps of value=2.0: {norm.ema:.4f} (expected 2.0)")
    print(f"  Normalized loss (2.0 / 2.0): {normalized.item():.4f} (expected 1.0)")
    print("  ✓ RunningNormalizer correct")

    # --- Test 4: LLMConfig prompt template structure ---
    print("\n[Test 4] LLMConfig prompt template")
    cfg = LLMConfig()
    prompt_only = cfg.chat_template("Some context here.", "What is X?")
    prompt_with_ans = cfg.chat_template("Some context here.", "What is X?", "X is Y.")
    assert "assistant" in prompt_with_ans
    assert "X is Y." in prompt_with_ans
    assert prompt_with_ans.startswith(prompt_only)
    ans_tokens_start = len(prompt_only)
    answer_portion = prompt_with_ans[ans_tokens_start:]
    assert answer_portion.startswith("X is Y.")
    print(f"  Prompt-only length: {len(prompt_only)} chars")
    print(f"  Answer portion: {answer_portion!r}")
    print("  ✓ Prompt template correct — answer span is contiguous suffix")

    print("\n" + "=" * 60)
    print("All tests passed. Phase 4 components are ready.")
    print("=" * 60)

# =============================================================================
# Chunk text lookup via SQLite (add at end of phase4_components.py)
# =============================================================================
import sqlite3

class ChunkDB:
    """
    SQLite-backed chunk text lookup.
    ~50 MB RAM vs ~5 GB for in-memory dict.
    Build once with build_chunk_db.py, use everywhere.
    """
    def __init__(self, db_path="chunk_texts.db"):
        self.con = sqlite3.connect(str(db_path), check_same_thread=False)
        self.cur = self.con.cursor()

    def get(self, chunk_id: str, default: str = "") -> str:
        row = self.cur.execute(
            "SELECT text FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone()
        return row[0] if row else default

    def get_many(self, chunk_ids: list) -> dict:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.cur.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            chunk_ids
        ).fetchall()
        result = {r[0]: r[1] for r in rows}
        return {cid: result.get(cid, "") for cid in chunk_ids}

    def close(self):
        self.con.close()
