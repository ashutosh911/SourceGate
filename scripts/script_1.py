"""
Script 1 (CORRECTED): Tokenize chunks with strict schema validation.
"""
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_LEN = 512
SHARD_SIZE = 100_000

CHUNKS_DIR = Path("chunks_by_source")
OUT_DIR = Path("tokenized")
OUT_DIR.mkdir(exist_ok=True)

SOURCES = ["nq", "triviaqa", "ott", "tat", "kg"]

print(f"Loading tokenizer: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(texts):
    enc = tok(texts, padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int8)


def save_shard(out_path, ids, masks, chunk_ids):
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "input_ids.npy", np.vstack(ids))
    np.save(out_path / "attention_mask.npy", np.vstack(masks))
    np.save(out_path / "chunk_ids.npy", np.array(chunk_ids, dtype=object))


def process_source(src):
    in_path = CHUNKS_DIR / f"{src}.jsonl"
    src_out = OUT_DIR / src
    src_out.mkdir(exist_ok=True)
    n_total = sum(1 for _ in open(in_path))
    print(f"  [{src}] {n_total:,} chunks total")

    buffer_texts, buffer_ids = [], []
    shard_ids, shard_masks, shard_cids = [], [], []
    shard_idx = 0
    BATCH = 256

    with open(in_path) as f:
        for i, line in enumerate(tqdm(f, total=n_total, desc=src)):
            obj = json.loads(line)
            text = obj["text"]  # strict — fail loudly if missing
            assert isinstance(text, str), f"line {i}: text is {type(text).__name__}"
            buffer_texts.append(text)
            buffer_ids.append(obj["id"])

            if len(buffer_texts) == BATCH:
                ids, masks = tokenize_batch(buffer_texts)
                shard_ids.append(ids); shard_masks.append(masks)
                shard_cids.extend(buffer_ids)
                buffer_texts, buffer_ids = [], []

                if len(shard_cids) >= SHARD_SIZE:
                    save_shard(src_out / f"shard_{shard_idx:04d}",
                               shard_ids, shard_masks, shard_cids)
                    shard_idx += 1
                    shard_ids, shard_masks, shard_cids = [], [], []

        if buffer_texts:
            ids, masks = tokenize_batch(buffer_texts)
            shard_ids.append(ids); shard_masks.append(masks)
            shard_cids.extend(buffer_ids)
        if shard_cids:
            save_shard(src_out / f"shard_{shard_idx:04d}",
                       shard_ids, shard_masks, shard_cids)
            shard_idx += 1

    print(f"  [{src}] DONE: {shard_idx} shards\n")


for src in SOURCES:
    print(f"=== Tokenizing {src} ===")
    process_source(src)

print("\n=== Summary ===")
for src in SOURCES:
    src_dir = OUT_DIR / src
    n = sum(np.load(s / "input_ids.npy", mmap_mode="r").shape[0]
            for s in src_dir.iterdir() if s.is_dir())
    print(f"  {src:10s} {n:>10,}")