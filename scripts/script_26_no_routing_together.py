"""
script_26_no_routing_together.py  —  No-routing baseline under Llama-3.3-70B

Retrieves top-10 chunks from ALL K=3 sources (union), concatenates them,
and generates answers via Together API. Matches script_10's infrastructure
exactly (same ChunkDB, same retriever, same prompt, same scoring).

USAGE:
  python script_26_no_routing_together.py
  python script_26_no_routing_together.py --n 20   # smoke test

OUTPUT:
  phase5_results/reader_scaling_together_predictions_no_routing.jsonl
  phase5_results/reader_scaling_together_no_routing_metrics.json
"""

import argparse, gc, json, os, re, string, time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ── Config (matches script_10 exactly) ───────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BGE_MODEL    = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_DIM    = 768
TOP_K        = 10
K            = 3

TEST_FILE    = "mmrag_test.json"
INDICES_DIR  = Path("faiss_indices")
RESULTS_DIR  = Path("phase5_results"); RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH   = RESULTS_DIR / "together_no_routing_cache.json"

SOURCE_TYPES    = ["text", "table", "kg"]
TYPE_TO_DATASETS = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}

TOGETHER_ENV_KEY = "TOGETHER_API_KEY"
TOGETHER_MODEL   = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TOGETHER_URL     = "https://api.together.xyz/v1/chat/completions"

# ── Prompt (matches script_10 exactly) ───────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the "
    "provided context. Be concise — give your best guess in a few words only."
)

# ── SQuAD scoring ─────────────────────────────────────────────────────────────
def normalize(s):
    def ra(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def ws(t): return ' '.join(t.split())
    def rp(t): return ''.join(c for c in t if c not in set(string.punctuation))
    return ws(ra(rp(s.lower())))

def f1_score(pred, gold_raw):
    from collections import Counter
    import ast
    if isinstance(gold_raw, str) and gold_raw.startswith("["):
        try: golds = ast.literal_eval(gold_raw)
        except: golds = [gold_raw]
    else:
        golds = [gold_raw] if isinstance(gold_raw, str) else gold_raw
    best = 0.0
    for g in golds:
        pt = normalize(pred).split(); gt = normalize(str(g)).split()
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n == 0: continue
        p = n / len(pt); r = n / len(gt)
        best = max(best, 2*p*r/(p+r))
    return best

def exact_match(pred, gold_raw):
    import ast
    if isinstance(gold_raw, str) and gold_raw.startswith("["):
        try: golds = ast.literal_eval(gold_raw)
        except: golds = [gold_raw]
    else:
        golds = [gold_raw] if isinstance(gold_raw, str) else gold_raw
    return float(any(normalize(pred) == normalize(str(g)) for g in golds))

# ── BGE encoder ───────────────────────────────────────────────────────────────
def encode_queries(queries):
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    bge = AutoModel.from_pretrained(BGE_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    prefixed = [QUERY_PREFIX + q for q in queries]
    out = np.empty((len(queries), EMBED_DIM), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, len(queries), 64), desc="BGE encode"):
            e = min(s+64, len(queries))
            enc = tok(prefixed[s:e], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(DEVICE)
            emb = bge(**enc).last_hidden_state[:, 0]
            out[s:e] = F.normalize(emb.float(), p=2, dim=1).cpu().numpy()
    del bge, tok; gc.collect(); torch.cuda.empty_cache()
    return out

# ── FAISS retrieval ───────────────────────────────────────────────────────────
def load_indices():
    indices, chunk_ids = {}, {}
    for ds in ["nq","triviaqa","ott","tat","kg"]:
        indices[ds]   = faiss.read_index(str(INDICES_DIR/ds/"index.faiss"))
        chunk_ids[ds] = np.load(INDICES_DIR/ds/"chunk_ids.npy", allow_pickle=True)
        print(f"  {ds}: {indices[ds].ntotal:,} vectors")
    return indices, chunk_ids

def search_union(q_emb, indices, chunk_ids, top_k=TOP_K):
    """Retrieve top_k chunks from ALL sources (union / no-routing)."""
    all_scores, all_cids = [], []
    q = q_emb.reshape(1,-1).astype(np.float32)
    for src, datasets in TYPE_TO_DATASETS.items():
        for ds in datasets:
            s, fids = indices[ds].search(q, top_k)
            cids = [chunk_ids[ds][i] for i in fids[0] if i >= 0]
            all_scores.extend(s[0][:len(cids)])
            all_cids.extend(cids)
    # Global top_k by score
    pairs = sorted(zip(all_scores, all_cids), reverse=True)[:top_k]
    return [c for _,c in pairs]

# ── SQLite chunk lookup (positional, matches script_25 fix) ───────────────────
import sqlite3 as _sql
CHUNK_DB   = Path("chunk_texts.db")
_pos_cache = {}

def _build_pos_cache(ds):
    if ds in _pos_cache: return
    id_path = INDICES_DIR / ds / "chunk_ids.npy"
    ordered_ids = list(np.load(id_path, allow_pickle=True))
    conn = _sql.connect(str(CHUNK_DB))
    id_to_text = {}
    BATCH = 500
    for s in range(0, len(ordered_ids), BATCH):
        batch = ordered_ids[s:s+BATCH]
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({','.join('?'*len(batch))})",
            batch).fetchall()
        id_to_text.update({r[0]:r[1] for r in rows})
    conn.close()
    _pos_cache[ds] = {cid: id_to_text.get(cid,"") for cid in ordered_ids}

def get_chunk_text(chunk_id):
    if chunk_id.startswith("nq_"):         ds = "nq"
    elif chunk_id.startswith("triviaqa_"): ds = "triviaqa"
    elif chunk_id.startswith("ott_"):      ds = "ott"
    elif chunk_id.startswith("tat_"):      ds = "tat"
    else:                                  ds = "kg"
    _build_pos_cache(ds)
    return _pos_cache[ds].get(chunk_id, "")

def format_chunks(cids):
    parts = [get_chunk_text(c) for c in cids]
    return "\n\n".join(p for p in parts if p)[:3000]

# ── Together generation ───────────────────────────────────────────────────────
RETRY_BASE = 2
MAX_RETRIES = 4

def generate_together(query, context, api_key):
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(TOGETHER_URL, headers=headers,
                              json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  API error: {e}")
                return ""
            time.sleep(RETRY_BASE * (2**attempt))
    return ""

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get(TOGETHER_ENV_KEY)
    if not api_key:
        print(f"ERROR: set {TOGETHER_ENV_KEY} environment variable")
        return

    print("Loading test data...")
    with open(TEST_FILE) as f:
        test_data = json.load(f)
    n = args.n or len(test_data)
    test_data = test_data[:n]

    print("Encoding queries...")
    query_embs = encode_queries([r["query"] for r in test_data])

    print("Loading FAISS indices...")
    indices, chunk_ids = load_indices()

    print("Pre-loading chunk DB...")
    for src, datasets in TYPE_TO_DATASETS.items():
        for ds in datasets:
            _build_pos_cache(ds)

    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        cache = json.load(open(CACHE_PATH))
        print(f"Loaded {len(cache)} cached entries")

    f1s, ems, predictions = [], [], []
    per_type_f1 = defaultdict(list)
    total_in, total_out = 0, 0
    t0 = time.time()

    for i, item in enumerate(tqdm(test_data, desc="no_routing")):
        key = str(i)
        if key in cache:
            pred = cache[key]
        else:
            cids = search_union(query_embs[i], indices, chunk_ids)
            context = format_chunks(cids)
            pred = generate_together(item["query"], context, api_key)
            cache[key] = pred
            if (i+1) % 50 == 0:
                with open(CACHE_PATH,"w") as f: json.dump(cache,f)
                elapsed = time.time()-t0
                eta = elapsed/(i+1)*(n-i-1)
                print(f"  [{i+1}/{n}] F1={np.mean(f1s):.4f} ETA={eta/60:.0f}min")

        f1 = f1_score(pred, item["answer"])
        em = exact_match(pred, item["answer"])
        oracle_t = SOURCE_TYPES[max(range(K), key=lambda j: [
            max(item["dataset_score"].get("nq",0),
                item["dataset_score"].get("triviaqa",0)),
            max(item["dataset_score"].get("ott",0),
                item["dataset_score"].get("tat",0)),
            item["dataset_score"].get("kg",0)][j])]

        f1s.append(f1); ems.append(em)
        per_type_f1[oracle_t].append(f1)
        predictions.append({"id":item.get("id",str(i)),
                             "query":item["query"], "gold":item["answer"],
                             "predicted":pred, "picked_type":"union",
                             "oracle_type":oracle_t, "f1":f1, "em":em})

    with open(CACHE_PATH,"w") as f: json.dump(cache,f)

    # Save predictions
    pred_path = RESULTS_DIR/"reader_scaling_together_predictions_no_routing.jsonl"
    with open(pred_path,"w") as f:
        for p in predictions: f.write(json.dumps(p)+"\n")

    f1_mean = float(np.mean(f1s))
    em_mean = float(np.mean(ems))
    metrics = {"method":"no_routing","reader":"llama_33_70b_instruct",
               "n":n,"f1_mean":f1_mean,"em_mean":em_mean,
               "per_type_f1":{k:float(np.mean(v)) for k,v in per_type_f1.items()}}
    with open(RESULTS_DIR/"reader_scaling_together_no_routing_metrics.json","w") as f:
        json.dump(metrics,f,indent=2)

    print(f"\n{'='*60}")
    print(f"NO-ROUTING  (Llama-3.3-70B, union retrieval)")
    print(f"{'='*60}")
    print(f"  F1 = {f1_mean:.4f}   EM = {em_mean:.4f}   n = {n}")
    elapsed = time.time()-t0
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"\nSaved predictions → {pred_path}")

if __name__ == "__main__":
    main()
