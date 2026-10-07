"""
script_7_evaluation.py (OPTIMIZED)

Phase 5: Final test-set evaluation.

Optimizations vs. naive version:
  1. Generation cache: (query_idx, picked_type) → (answer, nll). Many methods
     route the same query to the same source — we generate once, look up after.
  2. Skip NLL for baselines (random/majority/no_routing) — not used in any
     reported metric. NLL kept for oracle, phase3, phase4 (where it's meaningful).

Result: ~3-4 hours instead of ~11 for full test set.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import torch.nn as nn
import torch.nn.functional as F
import faiss
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
)
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3, SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS, EMBED_DIM,
)
from phase4_components import (
    LLMConfig, format_chunks, ChunkDB,
)


# =============================================================================
# Config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 10
MAX_NEW_TOKENS = 32
TEST_FILE = "mmrag_test.json"
INDICES_DIR = Path("faiss_indices")
PHASE3_CKPT_DIR = Path("checkpoints")
PHASE4_CKPT_DIR = Path("checkpoints_phase4")
RESULTS_DIR = Path("phase5_results"); RESULTS_DIR.mkdir(exist_ok=True)
GEN_CACHE_PATH = RESULTS_DIR / "generation_cache.json"

llm_cfg = LLMConfig(model_name=LLM_MODEL, max_seq_len=2048)

# Methods that need NLL (others get NaN to skip the second forward pass)
METHODS_NEEDING_NLL = {"oracle"}  # plus phase3_seedX and phase4_seedX, handled below


# =============================================================================
# Answer normalization (SQuAD-style)
# =============================================================================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
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
# Test-data loading
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
    out = []
    skipped = 0
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
# Multi-source retriever
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

    def search_all_union(self, query_embs_np, top_k):
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
# LLM
# =============================================================================
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


@torch.no_grad()
def generate_answer(llm, llm_tok, query, context, cfg):
    prompt = cfg.chat_template(context, query, answer=None)
    enc = llm_tok(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    if enc["input_ids"].size(1) > cfg.max_seq_len - MAX_NEW_TOKENS:
        keep = cfg.max_seq_len - MAX_NEW_TOKENS
        overflow = enc["input_ids"].size(1) - keep
        enc["input_ids"] = enc["input_ids"][:, overflow:]
        enc["attention_mask"] = enc["attention_mask"][:, overflow:]
    out = llm.generate(
        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=llm_tok.pad_token_id, eos_token_id=llm_tok.eos_token_id,
    )
    gen = out[0, enc["input_ids"].size(1):]
    text = llm_tok.decode(gen, skip_special_tokens=True).strip()
    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>")[0]
    del enc, out
    return text.strip()


@torch.no_grad()
def compute_nll(llm, llm_tok, query, context, gold, cfg):
    prompt_only = cfg.chat_template(context, query, answer=None)
    full = cfg.chat_template(context, query, answer=gold)
    p = llm_tok(prompt_only, add_special_tokens=False, return_tensors="pt").input_ids[0]
    f = llm_tok(full, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if f.size(0) > cfg.max_seq_len:
        overflow = f.size(0) - cfg.max_seq_len
        p = p[overflow:]; f = f[overflow:]
    plen = p.size(0)
    if plen >= f.size(0):
        return float("nan")
    ids = f.unsqueeze(0).to(DEVICE)
    labels = ids.clone()
    labels[0, :plen] = -100
    out = llm(input_ids=ids, labels=labels)
    loss = out.loss.item()
    del ids, labels, out
    return loss


# =============================================================================
# BGE
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
# Routing strategies
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
# Generation cache
# =============================================================================
class GenerationCache:
    """
    Caches (query_idx, picked_type) -> {answer, nll_or_None}.
    
    Across all 10 methods, ~3000 unique (query, source) pairs vs. ~12,860 naive
    generations. The cache is persisted to disk so a partial run can resume.
    """
    def __init__(self, path=GEN_CACHE_PATH):
        self.path = path
        self.cache = {}
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            self.cache = {tuple(json.loads(k)): v for k, v in raw.items()}
            print(f"  Loaded {len(self.cache):,} cached generations from {path}")

    def get(self, query_idx, picked_type):
        return self.cache.get((query_idx, picked_type))

    def put(self, query_idx, picked_type, answer, nll):
        self.cache[(query_idx, picked_type)] = {"answer": answer, "nll": nll}

    def has_nll(self, query_idx, picked_type):
        v = self.cache.get((query_idx, picked_type))
        return v is not None and v["nll"] is not None

    def save(self):
        raw = {json.dumps(list(k)): v for k, v in self.cache.items()}
        with open(self.path, "w") as f:
            json.dump(raw, f)


def get_or_generate(cache, query_idx, picked_type, query, context, gold,
                    llm, llm_tok, need_nll):
    """
    Returns (answer, nll). Generates and caches if missing.
    If need_nll=True and cache lacks nll, computes it and updates cache.
    """
    cached = cache.get(query_idx, picked_type)
    if cached is None:
        answer = generate_answer(llm, llm_tok, query, context, llm_cfg)
        nll = compute_nll(llm, llm_tok, query, context, gold, llm_cfg) if need_nll else None
        cache.put(query_idx, picked_type, answer, nll)
        return answer, nll
    # Already have answer; need NLL too?
    if need_nll and cached["nll"] is None:
        nll = compute_nll(llm, llm_tok, query, context, gold, llm_cfg)
        cache.put(query_idx, picked_type, cached["answer"], nll)
        return cached["answer"], nll
    return cached["answer"], cached["nll"]


# =============================================================================
# Per-method evaluation (cached)
# =============================================================================
def evaluate_method(method_name, records, query_embs, routing_decisions,
                    retriever, chunk_db, llm, llm_tok, cache,
                    need_nll, eval_n=None, save_predictions=True):
    n = len(records) if eval_n is None else min(eval_n, len(records))
    print(f"\n=== Evaluating: {method_name} (N={n}, NLL={need_nll}) ===")

    f1s, ems, nlls = [], [], []
    recall_at_picked = []
    per_type_f1 = defaultdict(list)
    predictions = []
    n_cached = 0

    t0 = time.time()
    for i in tqdm(range(n), desc=method_name):
        r = records[i]
        q_emb = query_embs[i:i+1]

        if method_name == "no_routing":
            cids = retriever.search_all_union(q_emb, TOP_K)[0]
            picked_type = "all"
        else:
            picked_idx = int(routing_decisions[i])
            picked_type = SOURCE_TYPES[picked_idx]
            cids = retriever.search_type(q_emb, picked_type, TOP_K)[0]

        chunks_dict = chunk_db.get_many(list(cids))
        chunks = [chunks_dict[c] for c in cids]
        context = format_chunks(chunks)

        # Generation (cached)
        cached_before = cache.get(i, picked_type)
        if cached_before is not None and (not need_nll or cached_before["nll"] is not None):
            n_cached += 1
        pred, nll = get_or_generate(
            cache, i, picked_type, r["query"], context, r["answer"],
            llm, llm_tok, need_nll,
        )

        f1 = f1_score(pred, r["answer"])
        em = exact_match(pred, r["answer"])
        oracle_t = SOURCE_TYPES[r["oracle_label"]]
        per_type_f1[oracle_t].append(f1)

        if method_name != "no_routing":
            relevant = set(cid for cid, s in r["relevant_chunks"].items() if s > 0)
            recall_at_picked.append(float(bool(set(cids.tolist()) & relevant)))

        f1s.append(f1); ems.append(em)
        if nll is not None and not np.isnan(nll):
            nlls.append(nll)
        predictions.append({
            "id": r["id"], "query": r["query"], "gold": r["answer"],
            "predicted": pred, "picked_type": picked_type,
            "oracle_type": oracle_t, "f1": f1, "em": em, "nll": nll,
        })

        if (i + 1) % 200 == 0:
            cache.save()  # periodic checkpoint

    cache.save()
    elapsed = time.time() - t0
    print(f"  {n} queries in {elapsed:.0f}s ({n/elapsed:.2f}/s, cached: {n_cached}/{n})")

    metrics = {
        "method": method_name, "n": n,
        "f1_mean": float(np.mean(f1s)),
        "em_mean": float(np.mean(ems)),
        "nll_mean": float(np.mean(nlls)) if nlls else float("nan"),
        "recall_at_picked": float(np.mean(recall_at_picked)) if recall_at_picked else float("nan"),
        "per_type_f1": {k: float(np.mean(v)) for k, v in per_type_f1.items()},
        "per_type_n": {k: len(v) for k, v in per_type_f1.items()},
    }
    if routing_decisions is not None:
        oracle = np.array([r["oracle_label"] for r in records[:n]])
        acc = float((routing_decisions[:n] == oracle).mean())
        per_type_acc = {}
        for j, t in enumerate(SOURCE_TYPES):
            mask = (oracle == j)
            if mask.sum() > 0:
                per_type_acc[t] = float((routing_decisions[:n][mask] == oracle[mask]).mean())
        macro = float(np.mean(list(per_type_acc.values()))) if per_type_acc else float("nan")
        metrics["routing_acc"] = acc
        metrics["routing_macro"] = macro
        metrics["routing_per_type_acc"] = per_type_acc

    print(f"  F1={metrics['f1_mean']:.4f}  EM={metrics['em_mean']:.4f}  "
          f"NLL={metrics['nll_mean']:.4f}")
    if "routing_acc" in metrics:
        print(f"  Routing acc={metrics['routing_acc']:.4f} macro={metrics['routing_macro']:.4f}")

    if save_predictions:
        out_path = RESULTS_DIR / f"predictions_{method_name}.jsonl"
        with open(out_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved predictions to {out_path}")

    return metrics


def aggregate_seeds(method_name, per_seed):
    keys = ["f1_mean", "em_mean", "nll_mean", "recall_at_picked",
            "routing_acc", "routing_macro"]
    agg = {"method": method_name, "n_seeds": len(per_seed)}
    for k in keys:
        vals = [m[k] for m in per_seed if k in m and not np.isnan(m[k])]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[k.replace("_mean", "") + "_std"] = float(np.std(vals))
    all_types = set()
    for m in per_seed:
        all_types.update(m.get("per_type_f1", {}).keys())
    agg["per_type_f1"] = {}
    for t in all_types:
        vals = [m["per_type_f1"][t] for m in per_seed if t in m.get("per_type_f1", {})]
        agg["per_type_f1"][t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Limit eval to first N test queries (for debugging)")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--phase4_seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--reset_cache", action="store_true",
                        help="Delete the generation cache before starting")
    args = parser.parse_args()

    if args.reset_cache and GEN_CACHE_PATH.exists():
        GEN_CACHE_PATH.unlink()
        print("Cache reset.")

    print(f"Phase 5 evaluation (optimized)")
    print(f"  Limit N: {args.n or 'full test set'}")
    print(f"  Phase 4 seeds: {args.phase4_seeds}")

    print("\nLoading test set...")
    test_records = load_test_records(TEST_FILE)

    print("Loading train set (for majority baseline)...")
    with open("mmrag_train.json") as f:
        train_data = json.load(f)
    train_records = []
    for item in train_data:
        oracle = oracle_type(item)
        if oracle is not None:
            train_records.append({"oracle_label": oracle})

    print("\nLoading components...")
    chunk_db = ChunkDB("chunk_texts.db")
    retriever = MultiSourceRetriever()
    bge, bge_tok = load_bge()
    llm, llm_tok = load_frozen_llm()

    print("\nEncoding test queries with BGE...")
    query_embs = encode_queries(bge, bge_tok, [r["query"] for r in test_records])
    del bge, bge_tok
    gc.collect(); torch.cuda.empty_cache()

    cache = GenerationCache()

    methods_to_run = args.methods or [
        "random", "majority", "oracle", "no_routing",
        "phase3", "phase4",
    ]

    all_metrics = {}

    # Baselines that DON'T need NLL — saves ~half the eval time on these
    if "random" in methods_to_run:
        decisions = routing_random(test_records, seed=42)
        all_metrics["random"] = evaluate_method(
            "random", test_records, query_embs, decisions,
            retriever, chunk_db, llm, llm_tok, cache,
            need_nll=False, eval_n=args.n,
        )

    if "majority" in methods_to_run:
        decisions = routing_majority(test_records, train_records)
        all_metrics["majority"] = evaluate_method(
            "majority", test_records, query_embs, decisions,
            retriever, chunk_db, llm, llm_tok, cache,
            need_nll=False, eval_n=args.n,
        )

    if "no_routing" in methods_to_run:
        all_metrics["no_routing"] = evaluate_method(
            "no_routing", test_records, query_embs, None,
            retriever, chunk_db, llm, llm_tok, cache,
            need_nll=False, eval_n=args.n,
        )

    # Methods that DO need NLL
    if "oracle" in methods_to_run:
        decisions = routing_oracle(test_records)
        all_metrics["oracle"] = evaluate_method(
            "oracle", test_records, query_embs, decisions,
            retriever, chunk_db, llm, llm_tok, cache,
            need_nll=True, eval_n=args.n,
        )

    if "phase3" in methods_to_run:
        per_seed = []
        for seed in args.phase4_seeds:
            ckpt_path = PHASE3_CKPT_DIR / f"sourceformer_k3_seed{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"  [skip] Phase 3 seed {seed}")
                continue
            print(f"\nLoading Phase 3 seed {seed}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            sf = SourceFormerK3().to(DEVICE)
            sf.load_state_dict(ckpt["state_dict"])
            decisions = routing_sourceformer(test_records, query_embs, sf)
            m = evaluate_method(
                f"phase3_seed{seed}", test_records, query_embs, decisions,
                retriever, chunk_db, llm, llm_tok, cache,
                need_nll=True, eval_n=args.n,
                save_predictions=(seed == args.phase4_seeds[0]),
            )
            per_seed.append(m)
            del sf
        all_metrics["phase3_per_seed"] = per_seed
        if per_seed:
            all_metrics["phase3"] = aggregate_seeds("phase3", per_seed)

    if "phase4" in methods_to_run:
        per_seed = []
        for seed in args.phase4_seeds:
            ckpt_path = PHASE4_CKPT_DIR / f"phase4_seed{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"  [skip] Phase 4 seed {seed}")
                continue
            print(f"\nLoading Phase 4 seed {seed}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            sf = SourceFormerK3().to(DEVICE)
            sf.load_state_dict(ckpt["state_dict"])
            decisions = routing_sourceformer(test_records, query_embs, sf)
            m = evaluate_method(
                f"phase4_seed{seed}", test_records, query_embs, decisions,
                retriever, chunk_db, llm, llm_tok, cache,
                need_nll=True, eval_n=args.n,
                save_predictions=(seed == args.phase4_seeds[0]),
            )
            per_seed.append(m)
            del sf
        all_metrics["phase4_per_seed"] = per_seed
        if per_seed:
            all_metrics["phase4"] = aggregate_seeds("phase4", per_seed)

    # Final report
    print("\n" + "=" * 80)
    print("PHASE 5 — TEST SET RESULTS")
    print("=" * 80)
    print(f"{'Method':<25} {'F1':>8} {'EM':>8} {'NLL':>8} {'R@picked':>10} "
          f"{'Acc':>8} {'Macro':>8}")
    print("-" * 80)

    rows = []
    for name in ["random", "majority", "oracle", "no_routing"]:
        if name in all_metrics:
            rows.append((name, all_metrics[name]))
    for name in ["phase3", "phase4"]:
        if name in all_metrics:
            rows.append((name, all_metrics[name]))

    for name, m in rows:
        f1 = m.get("f1_mean", float("nan"))
        em = m.get("em_mean", float("nan"))
        nll = m.get("nll_mean", float("nan"))
        rec = m.get("recall_at_picked", float("nan"))
        acc = m.get("routing_acc", float("nan"))
        mac = m.get("routing_macro", float("nan"))
        is_agg = "f1_std" in m
        if is_agg:
            f1_str = f"{f1:.3f}±{m['f1_std']:.3f}"
            em_str = f"{em:.3f}±{m['em_std']:.3f}"
            print(f"{name:<25} {f1_str:>14} {em_str:>14} "
                  f"{nll:>8.3f} {rec:>10.3f} {acc:>8.3f} {mac:>8.3f}")
        else:
            print(f"{name:<25} {f1:>8.3f} {em:>8.3f} {nll:>8.3f} "
                  f"{rec:>10.3f} {acc:>8.3f} {mac:>8.3f}")

    out_path = RESULTS_DIR / "phase5_metrics.json"
    with open(out_path, "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        json.dump(safe(all_metrics), f, indent=2)
    print(f"\nFull metrics saved to {out_path}")
    print(f"Generation cache: {GEN_CACHE_PATH} ({len(cache.cache):,} entries)")


if __name__ == "__main__":
    main()
