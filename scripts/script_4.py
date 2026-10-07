"""
Script 4 (CORRECTED): Load FAISS indices ONE AT A TIME to avoid RAM blowup.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import gc

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 512
BATCH = 64
KS = [1, 5, 20]

DEV_FILE = "mmrag_dev.json"
INDICES_DIR = Path("faiss_indices")
SOURCES = ["nq", "triviaqa", "ott", "tat", "kg"]

print(f"Loading {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE).eval()


@torch.inference_mode()
def encode_queries(queries):
    n = len(queries)
    out = np.empty((n, model.config.hidden_size), dtype=np.float32)
    prefixed = [QUERY_PREFIX + q for q in queries]
    for start in tqdm(range(0, n, BATCH), desc="encoding queries"):
        end = min(start + BATCH, n)
        enc = tok(prefixed[start:end], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        outputs = model(**enc)
        emb = outputs.last_hidden_state[:, 0]
        emb = F.normalize(emb, p=2, dim=1)
        out[start:end] = emb.cpu().numpy()
    return out


print(f"\nLoading {DEV_FILE}")
with open(DEV_FILE) as f:
    dev = json.load(f)
print(f"  {len(dev)} queries")

queries = [item["query"] for item in dev]
query_embeddings = encode_queries(queries)

# Free the encoder — we don't need it anymore for retrieval
del model, tok
gc.collect()
torch.cuda.empty_cache()
print("Encoder freed.\n")


def oracle_sources(item):
    scores = item["dataset_score"]
    max_s = max(scores.values())
    if max_s == 0:
        return []
    return [src for src, s in scores.items() if s == max_s]


queries_by_source = defaultdict(list)
for i, item in enumerate(dev):
    for src in oracle_sources(item):
        queries_by_source[src].append((i, item))


def relevant_chunks_in_source(item, src):
    prefix_map = {"nq": "nq_", "triviaqa": "triviaqa_", "ott": "ott_",
                  "tat": "tat_", "kg": ("m.", "g.")}
    pref = prefix_map[src]
    relevant = set()
    for cid, score in item["relevant_chunks"].items():
        if score <= 0:
            continue
        if isinstance(pref, tuple):
            if cid.startswith(pref):
                relevant.add(cid)
        elif cid.startswith(pref):
            relevant.add(cid)
    return relevant


results = {src: [] for src in SOURCES}
no_relevant_in_source = {src: 0 for src in SOURCES}
max_k = max(KS)

for src in SOURCES:
    pairs = queries_by_source[src]
    if not pairs:
        continue

    # Load this source's index ONLY when needed
    print(f"\nLoading FAISS index for {src}...")
    index = faiss.read_index(str(INDICES_DIR / src / "index.faiss"))
    cid_map = np.load(INDICES_DIR / src / "chunk_ids.npy", allow_pickle=True)
    print(f"  {index.ntotal:,} vectors loaded. Querying {len(pairs)} dev questions...")

    q_idxs = [p[0] for p in pairs]
    q_embs = query_embeddings[q_idxs]
    items = [p[1] for p in pairs]

    scores, faiss_ids = index.search(q_embs, max_k)

    for row, item in enumerate(items):
        retrieved_cids = [cid_map[idx] for idx in faiss_ids[row]]
        relevant = relevant_chunks_in_source(item, src)
        if not relevant:
            no_relevant_in_source[src] += 1
            continue
        recalls = {}
        for k in KS:
            top_k = set(retrieved_cids[:k])
            recalls[k] = float(len(top_k & relevant) > 0)
        results[src].append(recalls)

    # Free index before next source
    del index, cid_map, scores, faiss_ids
    gc.collect()
    print(f"  Released {src} index.")

# --- Report ---
print("\n" + "=" * 60)
print("RECALL@k BY SOURCE (oracle routing)")
print("=" * 60)
print(f"{'source':<10} {'N':>5} {'no_rel':>7}", end="")
for k in KS:
    print(f"  R@{k:<3}", end="")
print()
print("-" * 60)

agg = {k: [] for k in KS}
for src in SOURCES:
    n_eval = len(results[src])
    n_norel = no_relevant_in_source[src]
    if n_eval == 0:
        print(f"{src:<10} {0:>5} {n_norel:>7}  (no eval)")
        continue
    print(f"{src:<10} {n_eval:>5} {n_norel:>7}", end="")
    for k in KS:
        recalls = [r[k] for r in results[src]]
        mean_r = np.mean(recalls)
        agg[k].extend(recalls)
        print(f"  {mean_r:>5.3f}", end="")
    print()

print("-" * 60)
print(f"{'OVERALL':<10} {len(agg[KS[0]]):>5} {'':>7}", end="")
for k in KS:
    print(f"  {np.mean(agg[k]):>5.3f}", end="")
print()

with open("retrieval_sanity.json", "w") as f:
    json.dump({
        "per_source": {
            src: {
                "n_eval": len(results[src]),
                "n_no_relevant": no_relevant_in_source[src],
                **{f"recall@{k}": float(np.mean([r[k] for r in results[src]])) if results[src] else None
                   for k in KS},
            }
            for src in SOURCES
        },
        "overall": {f"recall@{k}": float(np.mean(agg[k])) for k in KS},
    }, f, indent=2)
print("\nSaved retrieval_sanity.json")

# --- Verdict ---
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
threshold = 0.60
all_pass = True
for src in SOURCES:
    if not results[src]:
        continue
    r5 = np.mean([r[5] for r in results[src]])
    status = "✓" if r5 >= threshold else "✗"
    print(f"  {status} {src:10s} recall@5 = {r5:.3f}  (threshold: {threshold:.2f})")
    if r5 < threshold:
        all_pass = False

if all_pass:
    print("\n✓ Retrieval is good enough. Proceed to Phase 3 (SourceFormer).")
else:
    print("\n✗ Retrieval is the bottleneck. Options:")
    print("    - Try a stronger encoder (E5-large-v2, BGE-large)")
    print("    - Check chunk quality on failing sources")
    print("    - Inspect failed queries to see if they're answerable at all")`