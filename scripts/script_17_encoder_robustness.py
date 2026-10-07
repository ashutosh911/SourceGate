"""
script_17_encoder_robustness.py — Retrieval-only encoder robustness (Option: 2nd encoder)

WHY THIS EXISTS
  The AE's "narrow evaluation" complaint includes "one encoder" (BGE-base-en-v1.5
  everywhere). Building a second full benchmark (WebQSP+HotpotQA) is weeks of
  work and out of scope for this revision. This script is the cheap alternative:
  swap the RETRIEVAL encoder for a second, architecturally different model
  (E5-base-v2) on the SAME mmRAG corpus and test set, and check whether
  retrieval quality (recall@picked) and downstream F1/EM under oracle /
  no_routing hold up. It does NOT retrain SourceFormer's router — the router
  was trained on BGE embeddings and swapping the encoder would require
  retraining it from scratch (a much larger job, out of scope here). So this
  answers "is retrieval quality an artifact of BGE specifically?" not "does the
  learned router generalize to other encoders?" — that's future work, and the
  writeup should state the scope honestly.

USAGE
  Step 1 (one-time, ~1-2 hours on an RTX 5070 Ti, ~692 docs/sec measured):
    python script_17_encoder_robustness.py --build_index
    python script_17_encoder_robustness.py --build_index --datasets kg tat ott  # subset

  Step 2 (eval, reuses the same local Llama-3.1-8B-4bit reader as script_7,
  no API cost; results are directly comparable to phase5_metrics.json):
    python script_17_encoder_robustness.py --n 20      # smoke test
    python script_17_encoder_robustness.py             # full 1286 queries

OUTPUT
  faiss_indices_e5/{ds}/index.faiss, chunk_ids.npy   (new indices, does not touch faiss_indices/)
  phase5_results/encoder_robustness_e5_metrics.json
  phase5_results/encoder_robustness_e5_cache.json     (generation cache, resumable)
"""

import argparse
import gc
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from script_7_evaluation import (
    load_frozen_llm, generate_answer, llm_cfg,
    load_test_records, oracle_type, f1_score, exact_match,
    TEST_FILE, MAX_NEW_TOKENS,
)
from sourceformer import SOURCE_TYPES, TYPE_TO_DATASETS
from phase4_components import format_chunks, ChunkDB

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E5_MODEL = "intfloat/e5-base-v2"
E5_DIM = 768
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
TOP_K = 10

CHUNK_DB = Path("chunk_texts.db")
E5_INDICES_DIR = Path("faiss_indices_e5")
RESULTS_DIR = Path("phase5_results"); RESULTS_DIR.mkdir(exist_ok=True)
DATASETS = ["nq", "triviaqa", "ott", "tat", "kg"]
DS_PREFIX = {"nq": "nq_%", "triviaqa": "triviaqa_%", "ott": "ott_%",
             "tat": "tat_%", "kg": "g.%"}


def e5_pool(last_hidden_states, attention_mask):
    """E5's required pooling: attention-mask-weighted mean over tokens.

    Unlike BGE (CLS-token pooling, last_hidden_state[:, 0]), E5 was trained
    with mean pooling and produces near-meaningless embeddings under CLS
    pooling -- this must not be copy-pasted from the BGE encode path.
    """
    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None].clamp(min=1)


# =============================================================================
# Step 1: build E5 indices (one-time). Mirrors build_text_index.py's approach
# but streams from chunk_texts.db (the SQLite chunk store already used
# everywhere else) instead of the raw processed_documents.json.
# =============================================================================
def build_e5_index(ds, batch_size=512):
    out_dir = E5_INDICES_DIR / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "index.faiss"
    ids_path = out_dir / "chunk_ids.npy"
    MIN_VALID_BYTES = 1024  # guards against empty/truncated files from a crashed prior run
    if (idx_path.exists() and ids_path.exists()
            and idx_path.stat().st_size > MIN_VALID_BYTES
            and ids_path.stat().st_size > 0):
        print(f"  [skip] {ds} E5 index already built ({idx_path})")
        return
    if idx_path.exists() or ids_path.exists():
        print(f"  [rebuild] {ds} had an incomplete/corrupt index — removing and rebuilding")
        idx_path.unlink(missing_ok=True)
        ids_path.unlink(missing_ok=True)

    print(f"\nBuilding E5 index for {ds}...")
    tok = AutoTokenizer.from_pretrained(E5_MODEL)
    model = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(DEVICE).eval()

    conn = sqlite3.connect(str(CHUNK_DB))
    cur = conn.execute("SELECT id, text FROM chunks WHERE id LIKE ? ORDER BY id",
                       (DS_PREFIX[ds],))
    count = conn.execute("SELECT COUNT(*) FROM chunks WHERE id LIKE ?",
                        (DS_PREFIX[ds],)).fetchone()[0]

    index = faiss.IndexFlatIP(E5_DIM)
    all_ids = []
    batch_ids, batch_texts = [], []

    @torch.no_grad()
    def flush():
        if not batch_ids:
            return
        prefixed = [E5_PASSAGE_PREFIX + (t or "") for t in batch_texts]
        enc = tok(prefixed, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
        emb = e5_pool(model(**enc).last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=1)
        index.add(emb.cpu().numpy().astype("float32"))
        all_ids.extend(batch_ids)

    t0 = time.time()
    pbar = tqdm(cur, total=count, desc=f"E5 encode {ds}")
    for cid, text in pbar:
        batch_ids.append(cid)
        batch_texts.append(text)
        if len(batch_ids) >= batch_size:
            flush()
            batch_ids, batch_texts = [], []
    flush()
    conn.close()

    faiss.write_index(index, str(idx_path))
    np.save(ids_path, np.array(all_ids, dtype=object))
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"  {ds}: {index.ntotal:,} vectors indexed in {elapsed/60:.1f} min "
          f"({index.ntotal/max(elapsed,1):.0f} docs/sec)")


# =============================================================================
# Step 2: retrieval + evaluation with the new E5 indices, same reader/harness
# as script_7 so numbers are directly comparable to phase5_metrics.json.
# =============================================================================
class E5Retriever:
    def __init__(self):
        self.ds_indices, self.ds_chunk_ids = {}, {}
        for ds in DATASETS:
            idx_path = E5_INDICES_DIR / ds / "index.faiss"
            if not idx_path.exists():
                raise SystemExit(
                    f"Missing {idx_path} — run with --build_index first "
                    f"(or --build_index --datasets {ds}).")
            self.ds_indices[ds] = faiss.read_index(str(idx_path))
            self.ds_chunk_ids[ds] = np.load(E5_INDICES_DIR / ds / "chunk_ids.npy",
                                            allow_pickle=True)

    def _search_ds(self, q_emb, ds, top_k):
        scores, fids = self.ds_indices[ds].search(q_emb, top_k)
        cid_map = self.ds_chunk_ids[ds]
        cids = np.array([[cid_map[i] for i in row] for row in fids], dtype=object)
        return scores, cids

    def search_type(self, q_emb, type_name, top_k):
        datasets = TYPE_TO_DATASETS[type_name]
        if len(datasets) == 1:
            _, cids = self._search_ds(q_emb, datasets[0], top_k)
            return cids
        all_scores, all_cids = [], []
        for ds in datasets:
            s, c = self._search_ds(q_emb, ds, top_k)
            all_scores.append(s); all_cids.append(c)
        merged_scores = np.concatenate(all_scores, axis=1)
        merged_cids = np.concatenate(all_cids, axis=1)
        top_idx = np.argsort(-merged_scores, axis=1)[:, :top_k]
        B = q_emb.shape[0]
        out = np.empty((B, top_k), dtype=object)
        for r in range(B):
            for c in range(top_k):
                out[r, c] = merged_cids[r, top_idx[r, c]]
        return out

    def search_all_union(self, q_emb, top_k):
        all_scores, all_cids = [], []
        for ds in DATASETS:
            s, c = self._search_ds(q_emb, ds, top_k)
            all_scores.append(s); all_cids.append(c)
        merged_scores = np.concatenate(all_scores, axis=1)
        merged_cids = np.concatenate(all_cids, axis=1)
        top_idx = np.argsort(-merged_scores, axis=1)[:, :top_k]
        B = q_emb.shape[0]
        out = np.empty((B, top_k), dtype=object)
        for r in range(B):
            for c in range(top_k):
                out[r, c] = merged_cids[r, top_idx[r, c]]
        return out


@torch.no_grad()
def encode_queries_e5(queries, batch=64):
    tok = AutoTokenizer.from_pretrained(E5_MODEL)
    model = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(DEVICE).eval()
    prefixed = [E5_QUERY_PREFIX + q for q in queries]
    out = np.empty((len(queries), E5_DIM), dtype=np.float32)
    for s in range(0, len(queries), batch):
        e = min(s + batch, len(queries))
        enc = tok(prefixed[s:e], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
        emb = e5_pool(model(**enc).last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=1)
        out[s:e] = emb.cpu().numpy()
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return out


class GenCache:
    def __init__(self, path):
        self.path = Path(path)
        self.cache = json.load(open(self.path)) if self.path.exists() else {}

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, val):
        self.cache[key] = val

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.cache, f)


def evaluate(method_name, records, cids_per_query, retriever, chunk_db,
             llm, llm_tok, cache, eval_n=None):
    n = len(records) if eval_n is None else min(eval_n, len(records))
    print(f"\n=== E5 / {method_name} | N={n} ===")
    f1s, ems, recalls = [], [], []
    per_type_f1 = defaultdict(list)
    t0 = time.time()

    for i in tqdm(range(n), desc=f"e5_{method_name}"):
        r = records[i]
        cids = cids_per_query[i]
        chunks_dict = chunk_db.get_many(list(cids))
        chunks = [chunks_dict[c] for c in cids]
        context = format_chunks(chunks)

        key = f"{method_name}:{i}"
        cached = cache.get(key)
        if cached is not None:
            pred = cached["answer"]
        else:
            pred = generate_answer(llm, llm_tok, r["query"], context, llm_cfg)
            cache.put(key, {"answer": pred})

        f1 = f1_score(pred, r["answer"])
        em = exact_match(pred, r["answer"])
        oracle_t = SOURCE_TYPES[r["oracle_label"]]
        per_type_f1[oracle_t].append(f1)
        relevant = set(cid for cid, s in r["relevant_chunks"].items() if s > 0)
        recalls.append(float(bool(set(cids.tolist()) & relevant)))
        f1s.append(f1); ems.append(em)

        if (i + 1) % 50 == 0:
            cache.save()
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"    [{i+1}/{n}] F1={np.mean(f1s):.4f} R@picked={np.mean(recalls):.4f} "
                  f"ETA={eta/60:.1f}min")

    cache.save()
    metrics = {
        "method": method_name, "n": n,
        "f1_mean": float(np.mean(f1s)), "em_mean": float(np.mean(ems)),
        "recall_at_picked": float(np.mean(recalls)),
        "per_type_f1": {k: float(np.mean(v)) for k, v in per_type_f1.items()},
    }
    print(f"  F1={metrics['f1_mean']:.4f} EM={metrics['em_mean']:.4f} "
          f"R@picked={metrics['recall_at_picked']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_index", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    if args.build_index:
        print("=" * 70)
        print("Building E5-base-v2 FAISS indices (retrieval-only robustness check)")
        print("=" * 70)
        for ds in args.datasets:
            build_e5_index(ds, batch_size=args.batch)
        print("\nDone. Re-run without --build_index to evaluate.")
        return

    print("=" * 70)
    print("Encoder robustness (E5-base-v2 retrieval) — oracle & no_routing")
    print("=" * 70)

    print("\nLoading test set...")
    test_records = load_test_records(TEST_FILE)

    print("Loading E5 FAISS indices + chunk DB...")
    retriever = E5Retriever()
    chunk_db = ChunkDB("chunk_texts.db")

    print("Encoding test queries with E5-base-v2...")
    query_embs = encode_queries_e5([r["query"] for r in test_records])

    n = len(test_records) if args.n is None else min(args.n, len(test_records))

    print("\nComputing oracle & no_routing retrieval (E5)...")
    oracle_cids, no_routing_cids = [], []
    for i in tqdm(range(n), desc="E5 retrieval"):
        q_emb = query_embs[i:i + 1]
        oracle_t = SOURCE_TYPES[test_records[i]["oracle_label"]]
        oracle_cids.append(retriever.search_type(q_emb, oracle_t, TOP_K)[0])
        no_routing_cids.append(retriever.search_all_union(q_emb, TOP_K)[0])

    print("\nLoading local Llama-3.1-8B-4bit reader (same as script_7, no API cost)...")
    llm, llm_tok = load_frozen_llm()
    cache = GenCache(RESULTS_DIR / "encoder_robustness_e5_cache.json")

    all_metrics = {}
    all_metrics["oracle"] = evaluate("oracle", test_records, oracle_cids,
                                     retriever, chunk_db, llm, llm_tok, cache, n)
    all_metrics["no_routing"] = evaluate("no_routing", test_records, no_routing_cids,
                                        retriever, chunk_db, llm, llm_tok, cache, n)

    out_path = RESULTS_DIR / "encoder_robustness_e5_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    bge_path = RESULTS_DIR / "phase5_metrics.json"
    bge = json.load(open(bge_path)) if bge_path.exists() else {}

    print("\n" + "=" * 70)
    print("BGE-base-en-v1.5 vs E5-base-v2 — retrieval-only robustness")
    print("=" * 70)
    print(f"{'method':<12}{'encoder':<10}{'F1':>8}{'EM':>8}{'R@picked':>10}")
    for m in ["oracle", "no_routing"]:
        if m in bge:
            print(f"{m:<12}{'BGE':<10}{bge[m]['f1_mean']:>8.4f}{bge[m]['em_mean']:>8.4f}"
                  f"{bge[m].get('recall_at_picked', float('nan')):>10.4f}")
        e = all_metrics[m]
        print(f"{m:<12}{'E5':<10}{e['f1_mean']:>8.4f}{e['em_mean']:>8.4f}"
              f"{e['recall_at_picked']:>10.4f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
