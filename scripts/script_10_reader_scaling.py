"""
script_10_reader_scaling.py — Reader-Scaling Experiment for SourceFormer

PURPOSE:
  Proves that SourceFormer's routing quality is independent of reader capacity.
  Same frozen router checkpoints (Phase 3 + Phase 4), same FAISS retrieval,
  but swap the reader from Llama 3.1 8B-4bit to:
    - Llama 3.3 70B Instruct (via Together.ai)
    - GPT-4o-mini (via OpenAI)

  If F1/EM improve while routing metrics stay constant, the reader was the
  bottleneck and the routing contribution is validated.

USAGE:
  # Llama 3.3 70B (cheapest — run this first)
  export TOGETHER_API_KEY="your-key"
  python script_10_reader_scaling.py --reader together

  # GPT-4o-mini
  export OPENAI_API_KEY="your-key-here"
  python script_10_reader_scaling.py --reader openai

  # Limit to N queries for cost estimation
  python script_10_reader_scaling.py --reader together --n 50

  # Run specific methods only
  python script_10_reader_scaling.py --reader together --methods oracle phase3

  # Resume after interruption (cache is auto-saved every 50 queries)
  python script_10_reader_scaling.py --reader together  # picks up from cache

COST ESTIMATES (full 1,286 test queries):
  Together.ai Llama 3.3 70B Turbo: ~$2-5
  OpenAI GPT-4o-mini:              ~$5-10

OUTPUT:
  phase5_results/reader_scaling_{reader}_metrics.json
  phase5_results/reader_scaling_{reader}_predictions_{method}.jsonl
  phase5_results/reader_scaling_comparison.json  (after both readers run)
"""

import os
import json
import re
import string
import time
import random
import argparse
import gc
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3, SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS, EMBED_DIM,
)
from phase4_components import format_chunks, ChunkDB

# Lazy imports — only load the SDK you need
requests = None
openai_mod = None


def _import_requests():
    global requests
    if requests is None:
        import requests as _r
        requests = _r


def _import_openai():
    global openai_mod
    if openai_mod is None:
        import openai as _o
        openai_mod = _o


def _import_local_llm():
    global AutoModelForCausalLM, AutoTokenizerLLM, BitsAndBytesConfig
    from transformers import AutoTokenizer as AutoTokenizerLLM
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# =============================================================================
# Config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 10
MAX_NEW_TOKENS = 32
TEST_FILE = "mmrag_test.json"
INDICES_DIR = Path("faiss_indices")
PHASE3_CKPT_DIR = Path("checkpoints")
PHASE4_CKPT_DIR = Path("checkpoints_phase4")
RESULTS_DIR = Path("phase5_results"); RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = "Answer the question based on the provided context. Be concise. Respond with ONLY the answer in a few words, no explanation. If unsure, give your best guess."

# Reader configs
READER_CONFIGS = {
    "together": {
        "name": "Llama-3.3-70B-Instruct",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "api_base": "https://api.together.xyz/v1/chat/completions",
        "env_key": "TOGETHER_API_KEY",
        "rpm_limit": 60,        # requests per minute (free tier)
        "cost_per_1m_input": 0.88,
        "cost_per_1m_output": 0.88,
    },
    "openai": {
        "name": "GPT-4o-mini",
        "model": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
        "rpm_limit": 500,
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
    },
    "llama8b": {
        "name": "Llama-3.1-8B-Instruct-4bit",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "api_base": None,
        "env_key": None,
        "rpm_limit": 100000,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
    },
}

# Retry config
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff


# =============================================================================
# API Reader — unified interface for Together.ai and OpenAI
# =============================================================================
class APIReader:
    """
    Calls a chat completion API to generate answers.

    Keeps the same prompt format as the local Llama 3.1 8B evaluation,
    but via API instead of local inference. No NLL computation — the
    scaling experiment only needs F1/EM to prove the reader-bottleneck
    argument.
    """

    def __init__(self, reader_key: str):
        cfg = READER_CONFIGS[reader_key]
        self.reader_key = reader_key
        self.model = cfg["model"]
        self.name = cfg["name"]
        self.api_base = cfg["api_base"]
        self.rpm_limit = cfg["rpm_limit"]
        self.cost_input = cfg["cost_per_1m_input"]
        self.cost_output = cfg["cost_per_1m_output"]

        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            raise RuntimeError(
                f"Set {cfg['env_key']} environment variable.\n"
                f"  Together.ai: https://api.together.xyz/settings/api-keys\n"
                f"  OpenAI:      https://platform.openai.com/api-keys"
            )
        self.api_key = api_key

        # Rate limiting
        self._min_interval = 60.0 / self.rpm_limit
        self._last_request = 0.0

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if reader_key == "openai":
            _import_openai()
        else:
            _import_requests()

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def generate(self, query: str, context: str) -> str:
        """Generate an answer given query + retrieved context."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                if self.reader_key == "openai":
                    return self._call_openai(messages)
                else:
                    return self._call_together(messages)
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES - 1 and any(
                    x in err_str.lower()
                    for x in ["rate limit", "429", "503", "timeout", "server error", "500"]
                ):
                    delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    print(f"    Retry {attempt+1}/{MAX_RETRIES} after {delay:.1f}s: {err_str[:80]}")
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries")

    def _call_together(self, messages: list) -> str:
        resp = requests.post(
            self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "stop": ["<|eot_id|>"],
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Together API {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        usage = data.get("usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"].strip()

    def _call_openai(self, messages: list) -> str:
        client = openai_mod.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )
        self.total_input_tokens += resp.usage.prompt_tokens
        self.total_output_tokens += resp.usage.completion_tokens
        return resp.choices[0].message.content.strip()

    def estimated_cost(self) -> float:
        return (
            self.total_input_tokens / 1e6 * self.cost_input
            + self.total_output_tokens / 1e6 * self.cost_output
        )


# =============================================================================
# Local Llama-3.1-8B-4bit reader — same SYSTEM_PROMPT/interface as APIReader,
# so evaluate_method() and everything downstream works unmodified. Exists to
# regenerate the 8B row of the reader-scaling table with the fixed prompt,
# matching how Oracle/Phase3/Phase4 were regenerated for together/openai.
# =============================================================================
LLAMA_TMPL = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
              "{system}<|eot_id|>\n"
              "<|start_header_id|>user<|end_header_id|>\n"
              "Context:\n{context}\n\nQuestion: {query}<|eot_id|>\n"
              "<|start_header_id|>assistant<|end_header_id|>\n")


class LocalLlamaReader:
    """Drop-in replacement for APIReader that runs Llama-3.1-8B-Instruct
    locally in 4-bit (NF4). Presents the same public interface (generate,
    reader_key, name, estimated_cost, total_input_tokens,
    total_output_tokens) so it needs no special-casing anywhere else in
    the script."""

    def __init__(self):
        _import_local_llm()
        cfg = READER_CONFIGS["llama8b"]
        self.reader_key = "llama8b"
        self.name = cfg["name"]
        self.cost_input = cfg["cost_per_1m_input"]
        self.cost_output = cfg["cost_per_1m_output"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        print("Loading Llama 3.1 8B Instruct (4-bit NF4)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_compute_dtype=torch.float16)
        self.tok = AutoTokenizerLLM.from_pretrained(cfg["model"])
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], quantization_config=bnb, device_map="auto")
        self.model.eval()

    @torch.no_grad()
    def generate(self, query: str, context: str) -> str:
        # Truncate context before templating — unlike the hosted together/
        # openai readers (large context windows), the local 4-bit 8B model
        # needs this to stay under the 1400-token safety cap below. Matches
        # script_25's local-8B path (context[:3000] chars).
        prompt = LLAMA_TMPL.format(system=SYSTEM_PROMPT, context=context[:3000], query=query)
        inp = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        self.total_input_tokens += inp.input_ids.shape[1]
        if inp.input_ids.shape[1] > 1400:
            return ""
        out = self.model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False, pad_token_id=self.tok.eos_token_id)
        new_tokens = out[0][inp.input_ids.shape[1]:]
        self.total_output_tokens += len(new_tokens)
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def compute_nll(self, query: str, context: str, gold: str) -> float:
        """Mean per-token negative log-likelihood of the gold answer span,
        conditioned on (context, query), under this frozen reader. Needed
        for Table 4's NLL column, which evaluate_method() does not compute
        (generate() only returns text, not logits). This requires direct
        model access -- not available for the API-based readers (OpenAI,
        Together), which don't expose token logprobs for forced
        continuations, so Table 4's NLL column is local-8B-only, matching
        every other NLL value already reported in that table.
        Same masking convention as PrefRAG-Conf's score_source() in
        script_12: prompt tokens masked with label=-100, only the answer
        span contributes to the loss.
        """
        prompt = LLAMA_TMPL.format(system=SYSTEM_PROMPT, context=context[:3000], query=query)
        full = prompt + " " + gold
        enc_prompt = self.tok(prompt, return_tensors="pt", add_special_tokens=False,
                               truncation=True, max_length=1400)
        enc_full = self.tok(full, return_tensors="pt", add_special_tokens=False,
                             truncation=True, max_length=1450).to(DEVICE)
        prompt_len = enc_prompt["input_ids"].shape[1]
        seq_len = enc_full["input_ids"].shape[1]
        if prompt_len >= seq_len:
            return float("nan")
        labels = enc_full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        out = self.model(**enc_full, labels=labels)
        return float(out.loss.item())

    def estimated_cost(self) -> float:
        return 0.0


# =============================================================================
# Answer normalization (SQuAD-style, identical to script_7)
# =============================================================================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, gold):
    p = normalize_answer(prediction).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


# =============================================================================
# Test data loading (identical to script_7)
# =============================================================================
def aggregated_type_scores(item):
    ts = {t: 0.0 for t in SOURCE_TYPES}
    for src, s in item["dataset_score"].items():
        if src in DATASET_TO_TYPE:
            ts[DATASET_TO_TYPE[src]] += float(s)
    return ts


def oracle_type(item):
    ts = aggregated_type_scores(item)
    total = sum(ts.values())
    if total == 0:
        return None
    return int(np.argmax([ts[t] for t in SOURCE_TYPES]))


def load_test_records(path):
    with open(path) as f:
        data = json.load(f)
    out, skipped = [], 0
    for item in data:
        oracle = oracle_type(item)
        if oracle is None:
            skipped += 1
            continue
        out.append({
            "id": item.get("id", ""),
            "query": item["query"],
            "answer": item["answer"],
            "oracle_label": oracle,
            "relevant_chunks": item["relevant_chunks"],
        })
    print(f"  Loaded {len(out):,} test queries ({skipped} skipped)")
    return out


# =============================================================================
# Multi-source retriever (identical to script_7)
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

    def search_type(self, query_embs_np, type_name, top_k):
        datasets = TYPE_TO_DATASETS[type_name]
        B = query_embs_np.shape[0]
        if len(datasets) == 1:
            ds = datasets[0]
            scores, faiss_ids = self.ds_indices[ds].search(query_embs_np, top_k)
            cid_map = self.ds_chunk_ids[ds]
            cids = np.empty((B, top_k), dtype=object)
            for r in range(B):
                for c in range(top_k):
                    cids[r, c] = cid_map[faiss_ids[r, c]]
            return cids
        all_scores, all_cids = [], []
        for ds in datasets:
            s, fids = self.ds_indices[ds].search(query_embs_np, top_k)
            cid_map = self.ds_chunk_ids[ds]
            c = np.empty((B, top_k), dtype=object)
            for r in range(B):
                for col in range(top_k):
                    c[r, col] = cid_map[fids[r, col]]
            all_scores.append(s); all_cids.append(c)
        merged_scores = np.concatenate(all_scores, axis=1)
        merged_cids = np.concatenate(all_cids, axis=1)
        top_indices = np.argsort(-merged_scores, axis=1)[:, :top_k]
        cids_merged = np.empty((B, top_k), dtype=object)
        for r in range(B):
            for c in range(top_k):
                cids_merged[r, c] = merged_cids[r, top_indices[r, c]]
        return cids_merged

    def search_union(self, query_embs_np, top_k):
        """No-routing baseline: search ALL five source indices in parallel
        and take the global top-k by score, same merge-then-rerank pattern
        as search_type's multi-dataset branch, just spanning every dataset
        instead of one type's constituent datasets."""
        B = query_embs_np.shape[0]
        all_scores, all_cids = [], []
        for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
            s, fids = self.ds_indices[ds].search(query_embs_np, top_k)
            cid_map = self.ds_chunk_ids[ds]
            c = np.empty((B, top_k), dtype=object)
            for r in range(B):
                for col in range(top_k):
                    c[r, col] = cid_map[fids[r, col]]
            all_scores.append(s); all_cids.append(c)
        merged_scores = np.concatenate(all_scores, axis=1)
        merged_cids = np.concatenate(all_cids, axis=1)
        top_indices = np.argsort(-merged_scores, axis=1)[:, :top_k]
        cids_merged = np.empty((B, top_k), dtype=object)
        for r in range(B):
            for c in range(top_k):
                cids_merged[r, c] = merged_cids[r, top_indices[r, c]]
        return cids_merged


# =============================================================================
# BGE encoder (identical to script_7)
# =============================================================================
def load_bge():
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    return model, tok


@torch.no_grad()
def encode_queries(bge, bge_tok, queries, batch=64):
    out = np.empty((len(queries), EMBED_DIM), dtype=np.float32)
    prefixed = [QUERY_PREFIX + q for q in queries]
    for s in range(0, len(queries), batch):
        e = min(s + batch, len(queries))
        enc = bge_tok(prefixed[s:e], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(DEVICE)
        emb = bge(**enc).last_hidden_state[:, 0]
        emb = F.normalize(emb.float(), p=2, dim=1)
        out[s:e] = emb.cpu().numpy()
    return out


# =============================================================================
# Routing strategies (identical to script_7)
# =============================================================================
def routing_random(records, seed):
    return np.random.default_rng(seed).integers(0, K, size=len(records))


def routing_majority(records, train_records):
    counts = Counter(r["oracle_label"] for r in train_records)
    majority = max(counts, key=counts.get)
    return np.full(len(records), majority)


def routing_oracle(records):
    return np.array([r["oracle_label"] for r in records])


def routing_sourceformer(records, query_embs, sf):
    sf.eval()
    with torch.no_grad():
        embs = torch.from_numpy(query_embs).float().to(DEVICE)
        return sf(embs).argmax(-1).cpu().numpy()


# =============================================================================
# Generation cache (reader-specific to avoid mixing with 8B results)
# =============================================================================
class GenerationCache:
    def __init__(self, path):
        self.path = path
        self.cache = {}
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            self.cache = {tuple(json.loads(k)): v for k, v in raw.items()}
            print(f"  Loaded {len(self.cache):,} cached generations from {path}")

    def get(self, query_idx, picked_type):
        return self.cache.get((query_idx, picked_type))

    def put(self, query_idx, picked_type, answer):
        self.cache[(query_idx, picked_type)] = {"answer": answer}

    def save(self):
        raw = {json.dumps(list(k)): v for k, v in self.cache.items()}
        with open(self.path, "w") as f:
            json.dump(raw, f)


# =============================================================================
# Per-method evaluation
# =============================================================================
def evaluate_method(method_name, records, query_embs, routing_decisions,
                    retriever, chunk_db, reader, cache,
                    eval_n=None, save_predictions=True, sample_seed=42):
    if eval_n is None:
        indices = list(range(len(records)))
    else:
        # mmrag_test.json is grouped by dataset (all "ott_*" records
        # first), so a positional records[:eval_n] slice is not a
        # representative sample -- confirmed on disk to produce ~50%
        # table / ~32% text / ~17% kg at eval_n=400, versus the true
        # ~30%/49%/22% split, which systematically depresses F1 since
        # table queries score much lower than text or kg (Table 8).
        # Sample random indices instead; records/query_embs themselves
        # are untouched, so this only affects which queries this run
        # covers, not their positions -- full runs and existing cache
        # entries (keyed by position) are unaffected.
        n = min(eval_n, len(records))
        indices = random.Random(sample_seed).sample(range(len(records)), n)
    n = len(indices)
    print(f"\n=== Evaluating: {method_name} | reader={reader.name} | N={n} ===")

    f1s, ems, nlls = [], [], []
    recall_at_picked = []
    per_type_f1 = defaultdict(list)
    predictions = []
    n_cached = 0
    supports_nll = hasattr(reader, "compute_nll")

    t0 = time.time()
    for loop_pos, i in enumerate(tqdm(indices, desc=method_name)):
        r = records[i]
        q_emb = query_embs[i:i+1]

        picked_idx = int(routing_decisions[i])
        picked_type = SOURCE_TYPES[picked_idx]
        cids = retriever.search_type(q_emb, picked_type, TOP_K)[0]

        chunks_dict = chunk_db.get_many(list(cids))
        chunks = [chunks_dict[c] for c in cids]
        context = format_chunks(chunks)

        # Check cache
        cached = cache.get(i, picked_type)
        if cached is not None:
            pred = cached["answer"]
            n_cached += 1
        else:
            pred = reader.generate(r["query"], context)
            cache.put(i, picked_type, pred)

        f1 = f1_score(pred, r["answer"])
        em = exact_match(pred, r["answer"])
        oracle_t = SOURCE_TYPES[r["oracle_label"]]
        per_type_f1[oracle_t].append(f1)

        relevant = set(cid for cid, s in r["relevant_chunks"].items() if s > 0)
        recall_at_picked.append(float(bool(set(cids.tolist()) & relevant)))

        if supports_nll:
            gold_str = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
            nlls.append(reader.compute_nll(r["query"], context, gold_str))

        f1s.append(f1); ems.append(em)
        predictions.append({
            "id": r["id"], "query": r["query"], "gold": r["answer"],
            "predicted": pred, "picked_type": picked_type,
            "oracle_type": oracle_t, "f1": f1, "em": em,
        })

        if (loop_pos + 1) % 50 == 0:
            cache.save()
            elapsed_so_far = time.time() - t0
            avg_per_q = elapsed_so_far / (loop_pos + 1)
            eta = avg_per_q * (n - loop_pos - 1)
            print(f"    [{loop_pos+1}/{n}] F1={np.mean(f1s):.4f} EM={np.mean(ems):.4f} "
                  f"cached={n_cached} cost=${reader.estimated_cost():.3f} "
                  f"ETA={eta/60:.1f}min")

    cache.save()
    elapsed = time.time() - t0
    print(f"  {n} queries in {elapsed:.0f}s ({n/max(elapsed,1):.2f}/s, cached: {n_cached}/{n})")
    print(f"  API cost so far: ${reader.estimated_cost():.3f} "
          f"({reader.total_input_tokens:,} in / {reader.total_output_tokens:,} out)")

    metrics = {
        "method": method_name,
        "reader": reader.name,
        "n": n,
        "f1_mean": float(np.mean(f1s)),
        "em_mean": float(np.mean(ems)),
        "recall_at_picked": float(np.mean(recall_at_picked)),
        "nll_mean": float(np.nanmean(nlls)) if nlls else None,
        "per_type_f1": {k: float(np.mean(v)) for k, v in per_type_f1.items()},
        "per_type_n": {k: len(v) for k, v in per_type_f1.items()},
    }

    oracle = np.array([r["oracle_label"] for r in records])[indices]
    decisions_sampled = np.asarray(routing_decisions)[indices]
    acc = float((decisions_sampled == oracle).mean())
    per_type_acc = {}
    for j, t in enumerate(SOURCE_TYPES):
        mask = (oracle == j)
        if mask.sum() > 0:
            per_type_acc[t] = float((decisions_sampled[mask] == oracle[mask]).mean())
    macro = float(np.mean(list(per_type_acc.values()))) if per_type_acc else float("nan")
    metrics["routing_acc"] = acc
    metrics["routing_macro"] = macro
    metrics["routing_per_type_acc"] = per_type_acc

    print(f"  F1={metrics['f1_mean']:.4f}  EM={metrics['em_mean']:.4f}  "
          f"R@picked={metrics['recall_at_picked']:.4f}")
    print(f"  Routing acc={acc:.4f} macro={macro:.4f}")

    if save_predictions:
        out_path = RESULTS_DIR / f"reader_scaling_{reader.reader_key}_predictions_{method_name}.jsonl"
        with open(out_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved predictions to {out_path}")

    return metrics


def evaluate_no_routing(records, query_embs, retriever, chunk_db, reader, cache,
                         eval_n=None, save_predictions=True, sample_seed=42):
    """No-routing baseline: union retrieval across all five source indices
    (search_union), no per-query source selection. Mirrors evaluate_method
    exactly except for retrieval (search_union vs. search_type) and metrics
    that don't apply without a routing decision (R@picked, routing acc/macro
    are not meaningful here, matching how the manuscript already describes
    this baseline for the together/openai readers)."""
    method_name = "no_routing"
    if eval_n is None:
        indices = list(range(len(records)))
    else:
        # Same fix as evaluate_method: mmrag_test.json is grouped by
        # dataset, so a positional slice is not representative.
        n = min(eval_n, len(records))
        indices = random.Random(sample_seed).sample(range(len(records)), n)
    n = len(indices)
    print(f"\n=== Evaluating: {method_name} | reader={reader.name} | N={n} ===")

    f1s, ems = [], []
    per_type_f1 = defaultdict(list)
    predictions = []
    n_cached = 0

    t0 = time.time()
    for loop_pos, i in enumerate(tqdm(indices, desc=method_name)):
        r = records[i]
        q_emb = query_embs[i:i+1]

        cids = retriever.search_union(q_emb, TOP_K)[0]

        chunks_dict = chunk_db.get_many(list(cids))
        chunks = [chunks_dict[c] for c in cids]
        context = format_chunks(chunks)

        # Cache key uses "union" in place of picked_type — distinguishes
        # these entries from any single-source-routed generation for the
        # same query index, since GenerationCache keys on (idx, picked_type).
        cached = cache.get(i, "union")
        if cached is not None:
            pred = cached["answer"]
            n_cached += 1
        else:
            pred = reader.generate(r["query"], context)
            cache.put(i, "union", pred)

        f1 = f1_score(pred, r["answer"])
        em = exact_match(pred, r["answer"])
        oracle_t = SOURCE_TYPES[r["oracle_label"]]
        per_type_f1[oracle_t].append(f1)

        f1s.append(f1); ems.append(em)
        predictions.append({
            "id": r["id"], "query": r["query"], "gold": r["answer"],
            "predicted": pred, "picked_type": "union",
            "oracle_type": oracle_t, "f1": f1, "em": em,
        })

        if (loop_pos + 1) % 50 == 0:
            cache.save()
            elapsed_so_far = time.time() - t0
            avg_per_q = elapsed_so_far / (loop_pos + 1)
            eta = avg_per_q * (n - loop_pos - 1)
            print(f"    [{loop_pos+1}/{n}] F1={np.mean(f1s):.4f} EM={np.mean(ems):.4f} "
                  f"cached={n_cached} cost=${reader.estimated_cost():.3f} "
                  f"ETA={eta/60:.1f}min")

    cache.save()
    elapsed = time.time() - t0
    print(f"  {n} queries in {elapsed:.0f}s ({n/max(elapsed,1):.2f}/s, cached: {n_cached}/{n})")
    print(f"  API cost so far: ${reader.estimated_cost():.3f} "
          f"({reader.total_input_tokens:,} in / {reader.total_output_tokens:,} out)")

    metrics = {
        "method": method_name,
        "reader": reader.name,
        "n": n,
        "f1_mean": float(np.mean(f1s)),
        "em_mean": float(np.mean(ems)),
        "recall_at_picked": None,   # not applicable — union has no single "picked" source
        "routing_acc": None,        # not applicable — no routing decision made
        "routing_macro": None,
        "per_type_f1": {k: float(np.mean(v)) for k, v in per_type_f1.items()},
        "per_type_n": {k: len(v) for k, v in per_type_f1.items()},
    }

    print(f"  F1={metrics['f1_mean']:.4f}  EM={metrics['em_mean']:.4f}  (R@picked: n/a, union retrieval)")

    if save_predictions:
        out_path = RESULTS_DIR / f"reader_scaling_{reader.reader_key}_predictions_{method_name}.jsonl"
        with open(out_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved predictions to {out_path}")

    return metrics


def aggregate_seeds(method_name, reader_name, per_seed):
    keys = ["f1_mean", "em_mean", "nll_mean", "recall_at_picked", "routing_acc", "routing_macro"]
    agg = {"method": method_name, "reader": reader_name, "n_seeds": len(per_seed)}
    for k in keys:
        vals = [
    m[k] for m in per_seed
    if k in m and m[k] is not None and not np.isnan(m[k])
            ]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
    all_types = set()
    for m in per_seed:
        all_types.update(m.get("per_type_f1", {}).keys())
    agg["per_type_f1"] = {}
    for t in all_types:
        vals = [m["per_type_f1"][t] for m in per_seed if t in m.get("per_type_f1", {})]
        agg["per_type_f1"][t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


# =============================================================================
# Comparison table builder
# =============================================================================
def build_comparison(reader_key, new_metrics):
    """
    Loads the original 8B results and builds a side-by-side comparison.
    """
    orig_path = RESULTS_DIR / "phase5_metrics.json"
    comp = {"reader_scaling_experiment": True, "readers": {}}

    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        comp["readers"]["llama_3.1_8b_4bit"] = {
            "phase3": orig.get("phase3", {}),
            "phase4": orig.get("phase4", {}),
            "oracle": orig.get("oracle", {}),
        }
    else:
        print("  [warn] phase5_metrics.json not found — 8B baseline won't be in comparison")

    reader_label = READER_CONFIGS[reader_key]["name"].lower().replace(" ", "_").replace("-", "_").replace(".", "")
    comp["readers"][reader_label] = new_metrics

    # Print comparison table
    print("\n" + "=" * 90)
    print("READER SCALING COMPARISON")
    print("=" * 90)
    print(f"{'Reader':<30} {'Method':<12} {'F1':>8} {'EM':>8} {'R@picked':>10} "
          f"{'Macro':>8}")
    print("-" * 90)

    for reader_name, reader_data in comp["readers"].items():
        for method in ["no_routing", "random", "oracle", "phase3", "phase4"]:
            m = reader_data.get(method, {})
            if not m:
                continue
            f1 = m.get("f1_mean", float("nan"))
            em = m.get("em_mean", float("nan"))
            rec = m.get("recall_at_picked")
            mac = m.get("routing_macro")
            rec = float("nan") if rec is None else rec
            mac = float("nan") if mac is None else mac
            f1_std = m.get("f1_mean_std", m.get("f1_std", 0))
            if f1_std and f1_std > 0:
                f1_str = f"{f1:.3f}±{f1_std:.3f}"
            else:
                f1_str = f"{f1:.3f}"
            print(f"{reader_name:<30} {method:<12} {f1_str:>14} {em:>8.3f} "
                  f"{rec:>10.3f} {mac:>8.3f}")

    print("=" * 90)
    print("\nKey insight: Routing metrics (R@picked, Macro) should be IDENTICAL across")
    print("readers — they depend only on the frozen router. F1/EM should INCREASE with")
    print("stronger readers, confirming the reader-bottleneck hypothesis.\n")

    comp_path = RESULTS_DIR / "reader_scaling_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)
    print(f"  Comparison saved to {comp_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Reader-scaling experiment for SourceFormer")
    parser.add_argument("--reader", choices=["together", "openai", "llama8b"], required=True,
                        help="Which reader to use (llama8b runs locally, no API key needed)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit eval to first N test queries (for cost estimation)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: oracle phase3 phase4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Phase 3/4 seeds to evaluate")
    parser.add_argument("--reset_cache", action="store_true",
                        help="Delete the reader-specific cache before starting")
    parser.add_argument("--decisions_file", nargs="+", default=None,
                        help="Score one or more pre-computed routing-decision "
                             "arrays through the corrected reader pipeline "
                             "(F1/EM/NLL/recall@picked/per-type breakdown), "
                             "e.g. for Table 4's still-missing baselines. "
                             "Each entry is NAME:PATH.npy, where the .npy is "
                             "an (n_test,) int array of source-type indices "
                             "(0/1/2 = text/table/kg for K=3), one entry per "
                             "test query, in mmrag_test.json order -- the "
                             "same array shape/order already used to compute "
                             "that method's existing Acc/Macro columns. "
                             "Example: --decisions_file "
                             "prefrag_conf:phase5_results/prefrag_conf_decisions_k3.npy "
                             "mlp_hardce:phase5_results/mlp_hardce_decisions_k3.npy")
    args = parser.parse_args()

    reader_cfg = READER_CONFIGS[args.reader]
    cache_path = RESULTS_DIR / f"reader_scaling_{args.reader}_cache.json"

    if args.reset_cache and cache_path.exists():
        cache_path.unlink()
        print("Cache reset.")

    print(f"{'='*70}")
    print(f"Reader Scaling Experiment — SourceFormer")
    print(f"{'='*70}")
    print(f"  Reader:  {reader_cfg['name']} ({reader_cfg['model']})")
    print(f"  API:     {args.reader}")
    print(f"  N:       {args.n or 'full test set'}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Cache:   {cache_path}")

    # Initialize reader (local 8B, or API-based)
    reader = LocalLlamaReader() if args.reader == "llama8b" else APIReader(args.reader)
    cache = GenerationCache(cache_path)

    # Load test data
    print("\nLoading test set...")
    test_records = load_test_records(TEST_FILE)

    print("Loading train set (for majority baseline label)...")
    with open("mmrag_train.json") as f:
        train_data = json.load(f)
    train_records = []
    for item in train_data:
        oracle = oracle_type(item)
        if oracle is not None:
            train_records.append({"oracle_label": oracle})

    # Load retrieval components
    print("\nLoading FAISS indices...")
    chunk_db = ChunkDB("chunk_texts.db")
    retriever = MultiSourceRetriever()

    print("\nLoading BGE encoder...")
    bge, bge_tok = load_bge()
    print("Encoding test queries...")
    query_embs = encode_queries(bge, bge_tok, [r["query"] for r in test_records])
    del bge, bge_tok
    gc.collect(); torch.cuda.empty_cache()

    methods_to_run = args.methods or ["oracle", "phase3", "phase4"]
    all_metrics = {}

    # External pre-computed decisions — scores any already-trained/already-
    # computed routing method (PrefRAG-Conf, Logistic Regression, MLP-HardCE,
    # R³AG-RQ, R³AG-full, ...) through the corrected reader pipeline. Each
    # such method already has an Acc/Macro column in Table 4 from a prior
    # run, computed from a saved (n_test,) routing-decision array; this
    # reuses that same array to fill in the still-missing F1/EM/NLL/
    # recall@picked columns, via the exact same evaluate_method() path
    # already validated for oracle/phase3/phase4/random.
    if args.decisions_file:
        for entry in args.decisions_file:
            name, path = entry.split(":", 1)
            decisions = np.load(path)
            if len(decisions) != len(test_records):
                raise ValueError(
                    f"--decisions_file {name}: array length {len(decisions)} "
                    f"!= {len(test_records)} test records. Check the file is "
                    f"the correct one for this test set and K setting."
                )
            print(f"\nLoaded external decisions for '{name}' from {path} "
                  f"({len(decisions)} entries)")
            all_metrics[name] = evaluate_method(
                name, test_records, query_embs, decisions,
                retriever, chunk_db, reader, cache,
                eval_n=args.n,
            )

    # No-routing — union retrieval across all five source indices, no
    # per-query source selection. Not part of the default set (matches
    # oracle/phase3/phase4 default); pass --methods no_routing explicitly.
    if "no_routing" in methods_to_run:
        all_metrics["no_routing"] = evaluate_no_routing(
            test_records, query_embs, retriever, chunk_db, reader, cache,
            eval_n=args.n,
        )

    # Random — lower-bound baseline. Single fixed seed (42), matching the
    # convention already used for this row elsewhere in the paper (no ±
    # reported). Not part of the default set; pass --methods random explicitly.
    if "random" in methods_to_run:
        decisions = routing_random(test_records, seed=42)
        all_metrics["random"] = evaluate_method(
            "random", test_records, query_embs, decisions,
            retriever, chunk_db, reader, cache,
            eval_n=args.n,
        )

    # Oracle — upper bound with this reader
    if "oracle" in methods_to_run:
        decisions = routing_oracle(test_records)
        all_metrics["oracle"] = evaluate_method(
            "oracle", test_records, query_embs, decisions,
            retriever, chunk_db, reader, cache,
            eval_n=args.n,
        )

    # Phase 3 (supervised pretraining)
    if "phase3" in methods_to_run:
        per_seed = []
        for seed in args.seeds:
            ckpt_path = PHASE3_CKPT_DIR / f"sourceformer_k3_seed{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"  [skip] Phase 3 seed {seed} — {ckpt_path} not found")
                continue
            print(f"\nLoading Phase 3 seed {seed}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            sf = SourceFormerK3().to(DEVICE)
            sf.load_state_dict(ckpt["state_dict"])
            decisions = routing_sourceformer(test_records, query_embs, sf)
            m = evaluate_method(
                f"phase3_seed{seed}", test_records, query_embs, decisions,
                retriever, chunk_db, reader, cache,
                eval_n=args.n,
                save_predictions=(seed == args.seeds[0]),
            )
            per_seed.append(m)
            del sf
        all_metrics["phase3_per_seed"] = per_seed
        if per_seed:
            all_metrics["phase3"] = aggregate_seeds("phase3", reader.name, per_seed)

    # Phase 4 (joint training)
    if "phase4" in methods_to_run:
        per_seed = []
        for seed in args.seeds:
            ckpt_path = PHASE4_CKPT_DIR / f"phase4_seed{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"  [skip] Phase 4 seed {seed} — {ckpt_path} not found")
                continue
            print(f"\nLoading Phase 4 seed {seed}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            sf = SourceFormerK3().to(DEVICE)
            sf.load_state_dict(ckpt["state_dict"])
            decisions = routing_sourceformer(test_records, query_embs, sf)
            m = evaluate_method(
                f"phase4_seed{seed}", test_records, query_embs, decisions,
                retriever, chunk_db, reader, cache,
                eval_n=args.n,
                save_predictions=(seed == args.seeds[0]),
            )
            per_seed.append(m)
            del sf
        all_metrics["phase4_per_seed"] = per_seed
        if per_seed:
            all_metrics["phase4"] = aggregate_seeds("phase4", reader.name, per_seed)

    # Final summary
    print("\n" + "=" * 70)
    print(f"RESULTS — {reader.name}")
    print("=" * 70)
    print(f"{'Method':<25} {'F1':>8} {'EM':>8} {'NLL':>8} {'R@picked':>10} {'Macro':>8}")
    print("-" * 78)
    _fixed_order = ["no_routing", "random", "oracle", "phase3", "phase4"]
    # Exclude "*_per_seed" entries -- those store a list of per-seed metric
    # dicts (for std-dev computation), not a single metrics dict, and would
    # crash the .get() calls below if included here.
    _print_order = _fixed_order + [
        k for k in all_metrics if k not in _fixed_order and not k.endswith("_per_seed")
    ]
    for name in _print_order:
        m = all_metrics.get(name, {})
        if not m:
            continue
        f1 = m.get("f1_mean", float("nan"))
        em = m.get("em_mean", float("nan"))
        nll = m.get("nll_mean")
        nll_s = "---" if nll is None else f"{nll:.3f}"
        rec = m.get("recall_at_picked")
        mac = m.get("routing_macro")
        rec_s = "---" if rec is None else f"{rec:.3f}"
        mac_s = "---" if mac is None else f"{mac:.3f}"
        f1_std = m.get("f1_mean_std", 0)
        if f1_std:
            print(f"{name:<25} {f1:.3f}±{f1_std:.3f}  {em:.3f}  "
                  f"{nll_s:>8} {rec_s:>10} {mac_s:>8}")
        else:
            print(f"{name:<25} {f1:>8.3f} {em:>8.3f} {nll_s:>8} {rec_s:>10} {mac_s:>8}")

    print(f"\nTotal API cost: ${reader.estimated_cost():.3f}")
    print(f"  Input tokens:  {reader.total_input_tokens:,}")
    print(f"  Output tokens: {reader.total_output_tokens:,}")

    # Save metrics
    out_path = RESULTS_DIR / f"reader_scaling_{args.reader}_metrics.json"
    all_metrics["api_cost"] = {
        "total_usd": reader.estimated_cost(),
        "input_tokens": reader.total_input_tokens,
        "output_tokens": reader.total_output_tokens,
    }
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

    # Build comparison table
    build_comparison(args.reader, all_metrics)


if __name__ == "__main__":
    main()
