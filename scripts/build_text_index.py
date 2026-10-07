#!/usr/bin/env python3
"""
build_text_index.py — Phase 2, streaming DPR FAISS builder for mmRAG
- Streams processed_documents.json (3.2M chunks) without loading into RAM
- Filters to text sources only: nq_*, triviaqa_*, hotpot_*
- Encodes with DPR question encoder (facebook/dpr-question_encoder-single-nq-base)
- Builds GPU FAISS index, saves to text_index.faiss + id_map.json

Designed for 16GB VRAM / 32GB RAM machine.
"""

import os, json, argparse, time
from tqdm import tqdm

try:
    import ijson
except ImportError:
    raise SystemExit("pip install ijson")

try:
    import torch
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    import faiss
    import numpy as np
except ImportError:
    raise SystemExit("pip install torch transformers faiss-gpu sentence-transformers")

def is_text_doc(doc_id):
    return doc_id.startswith(('nq_','triviaqa_','hotpot_'))

def stream_text_docs(json_path):
    """Yield (id, text) for text-source docs only"""
    with open(json_path, 'rb') as f:
        for obj in ijson.items(f, 'item'):
            doc_id = obj['id']
            if is_text_doc(doc_id):
                yield doc_id, obj['text']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', default='processed_documents.json')
    parser.add_argument('--model', default='facebook/dpr-question_encoder-single-nq-base')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--out_index', default='text_index.faiss')
    parser.add_argument('--out_map', default='text_id_map.json')
    args = parser.parse_args()

    print(f"Loading DPR model {args.model}...")
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.model)
    encoder = DPRQuestionEncoder.from_pretrained(args.model)
    encoder.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    print(f"Using device: {device}")

    # First pass: count docs (for progress bar)
    print("Counting text documents (streaming)...")
    count = 0
    for _ in stream_text_docs(args.docs):
        count += 1
        if count % 200000 == 0:
            print(f"  ...{count:,}")
    print(f"Total text docs: {count:,}")

    dim = encoder.config.hidden_size  # 768 for DPR
    # Use GPU index if available
    if device == 'cuda':
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        index = faiss.IndexFlatIP(dim)

    id_map = []
    batch_ids, batch_texts = [], []

    def encode_batch(ids, texts):
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            inputs = {k:v.to(device) for k,v in inputs.items()}
            emb = encoder(**inputs).pooler_output  # [B,768]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().astype('float32')

    print("Encoding and indexing...")
    start = time.time()
    for doc_id, text in tqdm(stream_text_docs(args.docs), total=count):
        batch_ids.append(doc_id)
        batch_texts.append(text)
        if len(batch_ids) >= args.batch:
            embs = encode_batch(batch_ids, batch_texts)
            index.add(embs)
            id_map.extend(batch_ids)
            batch_ids, batch_texts = [], []
    
    if batch_ids:
        embs = encode_batch(batch_ids, batch_texts)
        index.add(embs)
        id_map.extend(batch_ids)

    # Save
    print(f"Saving FAISS index to {args.out_index}...")
    if device == 'cuda':
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, args.out_index)
    else:
        faiss.write_index(index, args.out_index)

    with open(args.out_map, 'w') as f:
        json.dump(id_map, f)
    
    elapsed = time.time() - start
    print(f"Done. Indexed {len(id_map):,} docs in {elapsed/60:.1f} min")
    print(f"Index size: {os.path.getsize(args.out_index)/1e9:.2f} GB")

if __name__ == '__main__':
    main()
