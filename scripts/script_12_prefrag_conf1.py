"""
script_12_prefrag_conf.py — PrefRAG-Conf Baseline for SourceFormer (ROCm edition)

Implements the staged source-selection paradigm of PrefRAG (Zhao et al., 2024).
For each query, top-k chunks are retrieved from all three sources; the source
whose context yields the highest mean token log-probability of the question
tokens (single forward pass, no generate) is selected as the routing decision.

RAM-efficient design:
  - Model loads directly to GPU (no CPU staging) → saves 16 GB RAM
  - Chunks read from SQLite on-demand (no preload) → saves ~4 GB RAM
  - BGE embeddings cached to disk, loaded once as numpy array

USAGE:
  python script_12_prefrag_conf.py --n 20      # smoke test
  python script_12_prefrag_conf.py             # full run
  python script_12_prefrag_conf.py --resume    # resume interrupted run
"""

import json
import argparse
import time
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# =============================================================================
# Config
# =============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BGE_NAME     = "BAAI/bge-base-en-v1.5"
LLM_NAME     = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TEST_FILE    = "mmrag_test.json"
INDICES_DIR  = Path("faiss_indices")
CHUNK_DB     = Path("chunk_texts.db")          # SQLite — read on-demand
CHUNKS_DIR   = Path("chunks_by_source")        # JSONL fallback
EMB_CACHE    = Path("query_emb_cache") / "test_embs.npy"
RESULTS_DIR  = Path("phase5_results");  RESULTS_DIR.mkdir(exist_ok=True)

SOURCE_TYPES_K3     = ["text", "table", "kg"]
TYPE_TO_DATASETS_K3 = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
SOURCE_IDX          = {t: i for i, t in enumerate(SOURCE_TYPES_K3)}

TOP_K         = 10
MAX_CTX_CHARS = 800   # keep tokenised length well under 1024
ENCODE_BATCH  = 64

# =============================================================================
# Label helpers
# =============================================================================
def hard_label_k3(item):
    s = item["dataset_score"]
    return int(np.argmax([
        s.get("nq", 0) + s.get("triviaqa", 0),
        s.get("ott", 0) + s.get("tat", 0),
        s.get("kg", 0),
    ]))

# =============================================================================
# BGE encoding — reuse cache if available
# =============================================================================
def load_or_encode_test(test_records):
    if EMB_CACHE.exists():
        embs = np.load(EMB_CACHE)
        if embs.shape[0] == len(test_records):
            print(f"  BGE cache hit: {EMB_CACHE}  shape={embs.shape}")
            return embs
        print(f"  Cache size mismatch; re-encoding.")

    print(f"  Encoding {len(test_records)} queries with BGE...")
    tok = AutoTokenizer.from_pretrained(BGE_NAME)
    bge = AutoModel.from_pretrained(BGE_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    queries = [QUERY_PREFIX + r["query"] for r in test_records]
    n   = len(queries)
    out = np.empty((n, 768), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, n, ENCODE_BATCH), desc="BGE encode"):
            e   = min(s + ENCODE_BATCH, n)
            enc = tok(queries[s:e], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(DEVICE)
            emb = bge(**enc).last_hidden_state[:, 0]
            out[s:e] = F.normalize(emb.float(), p=2, dim=1).cpu().numpy()
    del bge, tok
    torch.cuda.empty_cache()
    EMB_CACHE.parent.mkdir(exist_ok=True)
    np.save(EMB_CACHE, out)
    return out

# =============================================================================
# FAISS retrieval
# =============================================================================
def load_indices():
    idx = {}
    for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
        idx[ds] = faiss.read_index(str(INDICES_DIR / ds / "index.faiss"))
    return idx

def retrieve_chunk_ids(q_emb, ds, faiss_idx, k):
    _, ids = faiss_idx.search(q_emb[None].astype(np.float32), k)
    return ids[0].tolist()

# =============================================================================
# Chunk retrieval — on-demand from SQLite (no RAM preload)
# =============================================================================
_db_conn = None
_suffix_cache = {}   # ds_name -> {int_suffix: text}

def get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(str(CHUNK_DB), check_same_thread=False)
    return _db_conn

def warm_suffix_cache(ds_name):
    """Build int-suffix -> text map for one dataset (called once, ~200 MB per ds)."""
    if ds_name in _suffix_cache:
        return
    conn = get_db()
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE id LIKE ?",
        (ds_name + "_%",)
    ).fetchall()
    m = {}
    for row_id, row_text in rows:
        try:
            m[int(row_id.rsplit("_", 1)[-1])] = row_text
        except (ValueError, IndexError):
            pass
    _suffix_cache[ds_name] = m
    print(f"  Cached {len(m):,} chunks for {ds_name}")

def get_chunk_texts_sqlite(ds_name, chunk_ids):
    """Lookup chunks from in-memory suffix cache (warmed once at startup)."""
    if not chunk_ids:
        return []
    warm_suffix_cache(ds_name)
    m = _suffix_cache[ds_name]
    return [m[cid] for cid in chunk_ids if cid in m]

def get_chunk_texts_jsonl(ds_name, chunk_ids):
    """JSONL fallback: scan file and return only needed lines by position."""
    path = CHUNKS_DIR / f"{ds_name}.jsonl"
    needed = set(chunk_ids)
    texts = {}
    with open(path) as f:
        for line_no, line in enumerate(f):
            if line_no in needed:
                texts[line_no] = json.loads(line)["text"]
            if len(texts) == len(needed):
                break
    return [texts[i] for i in chunk_ids if i in texts]

def get_chunk_texts(ds_name, chunk_ids):
    if CHUNK_DB.exists():
        return get_chunk_texts_sqlite(ds_name, chunk_ids)
    return get_chunk_texts_jsonl(ds_name, chunk_ids)

def retrieve_context(q_emb, source_type, faiss_indices, k):
    texts = []
    for ds in TYPE_TO_DATASETS_K3[source_type]:
        ids   = retrieve_chunk_ids(q_emb, ds, faiss_indices[ds], k)
        texts += get_chunk_texts(ds, ids)
    return "\n\n".join(texts)[:MAX_CTX_CHARS]

# =============================================================================
# LLM — load directly to GPU
# =============================================================================
def load_llm():
    print(f"\nLoading {LLM_NAME} → GPU (fp16)...")
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: "14GiB", "cpu": "4GiB"},  # spill tail layers only
        low_cpu_mem_usage=True,
    )
    llm.eval()
    vram = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  LLM loaded. VRAM used: {vram:.1f} GB")
    return tok, llm

# =============================================================================
# PrefRAG-Conf scoring — single forward pass, no generate()
# =============================================================================
def score_source(query, context, llm_tok, llm):
    """
    Mean token log-probability of the question tokens conditioned on context.
    Higher = model finds the question more predictable given this context.
    No generate() call → no KV cache → safe on 16 GB VRAM.
    """
    prefix = (
        "Answer the following question using only the provided context. "
        "Be concise.\n\nContext:\n" + context + "\n\nQuestion: "
    )
    full = prefix + query + "\n\nAnswer:"

    enc_prefix = llm_tok(prefix, return_tensors="pt",
                         truncation=True, max_length=900)
    enc_full   = llm_tok(full,   return_tensors="pt",
                         truncation=True, max_length=1024).to(DEVICE)

    prefix_len = enc_prefix["input_ids"].shape[1]
    seq_len    = enc_full["input_ids"].shape[1]
    if prefix_len >= seq_len:
        return -1e9

    labels = enc_full["input_ids"].clone()
    labels[:, :prefix_len] = -100   # score only question tokens

    with torch.no_grad():
        out = llm(**enc_full, labels=labels)

    return -out.loss.item()   # negate NLL → higher = better

# =============================================================================
# Evaluation
# =============================================================================
def eval_routing(preds, labels):
    acc = float((preds == labels).mean())
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES_K3):
        mask = (labels == idx)
        per_type[t] = float((preds[mask] == idx).mean()) if mask.sum() else float("nan")
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return acc, macro, per_type

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,  default=None)
    parser.add_argument("--top_k",  type=int,  default=TOP_K)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("PrefRAG-Conf  (staged re-ranking, no training, no gold labels)")
    print("=" * 70)

    with open(TEST_FILE) as f:
        test_data = json.load(f)
    if args.n:
        test_data = test_data[:args.n]
    print(f"Test queries: {len(test_data)}")

    # Embeddings
    test_embs = load_or_encode_test([{"query": t["query"]} for t in test_data])

    # FAISS
    print("\nLoading FAISS indices...")
    faiss_indices = load_indices()

    # Warm chunk cache — loads id->text maps for all 5 datasets into RAM.
    # Each dataset map is ~50-200 MB (just strings, no tensors).
    # Total: ~500 MB RAM, done before LLM loads to GPU.
    if CHUNK_DB.exists():
        print("\nWarming chunk cache from SQLite...")
        for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
            warm_suffix_cache(ds)

    # LLM — goes straight to GPU, no CPU staging
    llm_tok, llm = load_llm()

    # Resume
    results_path = RESULTS_DIR / "prefrag_conf_results.json"
    results, done_ids = [], set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r["query_idx"] for r in results}
        print(f"  Resuming from {len(done_ids)} completed queries.")

    print(f"\nRouting {len(test_data)} queries (top_k={args.top_k})...")
    t0 = time.time()

    for i, item in enumerate(tqdm(test_data, desc="PrefRAG-Conf")):
        if i in done_ids:
            continue

        query      = item["query"]
        true_label = hard_label_k3(item)
        q_emb      = test_embs[i]

        source_scores = {}
        for src in SOURCE_TYPES_K3:
            context = retrieve_context(q_emb, src, faiss_indices, args.top_k)
            source_scores[src] = score_source(query, context, llm_tok, llm)

        pred_src   = max(source_scores, key=source_scores.get)
        pred_label = SOURCE_IDX[pred_src]

        results.append({
            "query_idx":     i,
            "query":         query,
            "true_label":    int(true_label),
            "true_source":   SOURCE_TYPES_K3[true_label],
            "pred_label":    int(pred_label),
            "pred_source":   pred_src,
            "source_scores": source_scores,
            "correct":       int(pred_label == true_label),
        })

        if (i + 1) % 50 == 0:
            preds  = np.array([r["pred_label"] for r in results])
            labels = np.array([r["true_label"] for r in results])
            acc, macro, pt = eval_routing(preds, labels)
            elapsed = time.time() - t0
            eta     = elapsed / len(results) * (len(test_data) - len(results))
            print(f"  [{i+1}/{len(test_data)}] acc={acc:.3f} macro={macro:.3f} "
                  f"text={pt.get('text',0):.3f} table={pt.get('table',0):.3f} "
                  f"kg={pt.get('kg',0):.3f}  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    preds  = np.array([r["pred_label"] for r in results])
    labels = np.array([r["true_label"] for r in results])
    acc, macro, per_type = eval_routing(preds, labels)

    summary = {
        "method":        "PrefRAG-Conf",
        "description":   "Staged re-ranking via question-token log-probability (no training, no gold labels)",
        "n_queries":     len(results),
        "top_k":         args.top_k,
        "routing_acc":   acc,
        "routing_macro": macro,
        "routing_per_type_acc": per_type,
    }
    with open(RESULTS_DIR / "prefrag_conf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(RESULTS_DIR / "prefrag_conf_decisions_k3.npy",
            np.array([r["pred_label"] for r in results], dtype=np.int32))

    print(f"\n{'='*70}")
    print(f"PrefRAG-CONF RESULTS  (K=3, test set, n={len(results)})")
    print(f"{'='*70}")
    print(f"  acc={acc:.4f}  macro={macro:.4f}")
    print(f"  text={per_type.get('text',float('nan')):.3f}  "
          f"table={per_type.get('table',float('nan')):.3f}  "
          f"kg={per_type.get('kg',float('nan')):.3f}")
    print(f"\n  Comparison (K=3 test):")
    print(f"    Random:          macro=0.337")
    print(f"    BGE-confidence:  macro=0.661")
    print(f"    MLP-HardCE:      macro=0.702")
    print(f"    SF Phase 3:      macro=0.737")
    print(f"    SF Phase 4:      macro=0.717")
    print(f"\n  Saved to phase5_results/")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
