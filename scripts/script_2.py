"""
Script 2 (CORRECTED): Encode tokenized chunks with BGE.
Resilient to missing directories from prior runs.
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel
from tqdm import tqdm
import time

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOK_DIR = Path("tokenized")
OUT_DIR = Path("embeddings")
OUT_DIR.mkdir(exist_ok=True)

SOURCES = ["tat", "ott", "triviaqa", "nq", "kg"]

print(f"Loading {MODEL_NAME} on {DEVICE}")
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval().to(DEVICE)
EMBED_DIM = model.config.hidden_size
print(f"Embed dim: {EMBED_DIM}\n")


@torch.inference_mode()
def encode_shard(input_ids, attention_mask):
    n = input_ids.shape[0]
    out = np.empty((n, EMBED_DIM), dtype=np.float16)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        ids_batch = torch.from_numpy(input_ids[start:end]).long().to(DEVICE, non_blocking=True)
        mask_batch = torch.from_numpy(attention_mask[start:end]).long().to(DEVICE, non_blocking=True)
        outputs = model(input_ids=ids_batch, attention_mask=mask_batch)
        embeds = outputs.last_hidden_state[:, 0]
        embeds = F.normalize(embeds, p=2, dim=1)
        out[start:end] = embeds.cpu().numpy().astype(np.float16)
    return out


def process_source(src):
    src_tok = TOK_DIR / src
    if not src_tok.exists():
        print(f"  [skip] {src}: no tokenized data found at {src_tok}")
        return

    src_out = OUT_DIR / src
    src_out.mkdir(parents=True, exist_ok=True)  # ensure parent + dir both exist

    shards = sorted(p for p in src_tok.iterdir() if p.is_dir())
    print(f"=== {src}: {len(shards)} shards ===")

    for shard_dir in shards:
        out_shard = src_out / shard_dir.name
        out_emb = out_shard / "embeddings.npy"
        out_cids = out_shard / "chunk_ids.npy"

        if out_emb.exists() and out_cids.exists():
            print(f"  [skip] {shard_dir.name} already encoded")
            continue

        out_shard.mkdir(parents=True, exist_ok=True)

        input_ids = np.load(shard_dir / "input_ids.npy", mmap_mode="r")
        attention_mask = np.load(shard_dir / "attention_mask.npy", mmap_mode="r")
        chunk_ids = np.load(shard_dir / "chunk_ids.npy", allow_pickle=True)

        n = input_ids.shape[0]
        t0 = time.time()
        embeddings = encode_shard(input_ids, attention_mask)
        elapsed = time.time() - t0
        rate = n / elapsed
        print(f"  [{shard_dir.name}] {n:,} chunks in {elapsed:.1f}s ({rate:.0f}/s)")

        np.save(out_emb, embeddings)
        np.save(out_cids, chunk_ids)

    print()


for src in SOURCES:
    process_source(src)

# Summary — robust to missing directories
print("\n=== Encoding summary ===")
total_chunks = 0
total_mb = 0
for src in SOURCES:
    src_dir = OUT_DIR / src
    if not src_dir.exists():
        print(f"  {src:10s} (no output)")
        continue
    n_shards = sum(1 for _ in src_dir.iterdir() if _.is_dir())
    n_chunks = 0
    size_b = 0
    for shard in src_dir.iterdir():
        if not shard.is_dir():
            continue
        emb_path = shard / "embeddings.npy"
        if emb_path.exists():
            emb = np.load(emb_path, mmap_mode="r")
            n_chunks += emb.shape[0]
        for f in shard.iterdir():
            size_b += f.stat().st_size
    total_chunks += n_chunks
    total_mb += size_b / 1e6
    print(f"  {src:10s} {n_shards:>3d} shards, {n_chunks:>10,} chunks, {size_b/1e6:>7.1f} MB")

print(f"\n  TOTAL      {total_chunks:>10,} chunks, {total_mb:>7.1f} MB on disk")