"""
script_43_prefrag_scores_dev.py  --  PrefRAG-Conf question-token scores on the
                                     DEV split

WHY THIS IS THE BOTTLENECK
  The one significant positive result we have -- that reader likelihood adds
  +0.035 macro beyond SourceGate's own logits (CI [+0.011, +0.059], p=0.003) --
  was measured by cross-validation ON the test set, because no dev-split
  likelihood scores exist.  Same for the gated hybrid, whose threshold was
  chosen on test.  Neither can be published in that form.

  With dev scores, every combiner and every threshold is fitted on dev and
  evaluated once on a frozen test set, which is the protocol the claim needs.

  Scores use exactly script_12's corrected pipeline: chunk text fetched by id
  from chunk_ids.npy (not by FAISS position), one merged top-10 per source
  type, question tokens only.

USAGE
  conda activate chestx && python script_43_prefrag_scores_dev.py [--n N]

OUTPUT
  phase5_results/prefrag_conf_dev_scores.npz   scores (N,3), gold (N,), hit (N,3)
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import script_12_prefrag_conf as S12

SOURCES = ["text", "table", "kg"]
DEV_FILE = "mmrag_dev.json"
OUT = Path("phase5_results") / "prefrag_conf_dev_scores.npz"
DEV_EMB = Path("query_emb_cache") / "dev_embs_prefrag.npy"


def encode_queries(queries):
    """BGE encode with script_12's prefix and L2 normalisation, to our own path."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    if DEV_EMB.exists():
        e = np.load(DEV_EMB)
        if e.shape[0] == len(queries):
            print(f"  dev embedding cache hit: {DEV_EMB} {e.shape}")
            return e.astype(np.float32)

    tok = AutoTokenizer.from_pretrained(S12.BGE_NAME)
    bge = AutoModel.from_pretrained(S12.BGE_NAME,
                                    torch_dtype=torch.float16).to(S12.DEVICE).eval()
    qs = [S12.QUERY_PREFIX + q for q in queries]
    out = np.empty((len(qs), 768), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, len(qs), S12.ENCODE_BATCH), desc="BGE encode (dev)"):
            e = min(s + S12.ENCODE_BATCH, len(qs))
            enc = tok(qs[s:e], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(S12.DEVICE)
            out[s:e] = F.normalize(bge(**enc).last_hidden_state[:, 0].float(),
                                   p=2, dim=1).cpu().numpy()
    del bge, tok
    torch.cuda.empty_cache()
    DEV_EMB.parent.mkdir(exist_ok=True)
    np.save(DEV_EMB, out)
    print(f"  saved dev embeddings -> {DEV_EMB}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    dev = json.load(open(DEV_FILE))
    n = len(dev) if args.n is None else min(args.n, len(dev))
    dev = dev[:n]
    gold = np.array([S12.hard_label_k3(d) for d in dev])
    print(f"dev queries: {n}   gold counts "
          f"{dict(zip(SOURCES, [int((gold == j).sum()) for j in range(3)]))}")

    # Encode locally, NOT via S12.load_or_encode_test: that helper writes its
    # result to EMB_CACHE, which is query_emb_cache/test_embs.npy.  Calling it
    # with 766 dev records would miss the 1286-row cache and then overwrite the
    # test embeddings with dev ones, silently corrupting every downstream
    # evaluation that reads that file.  Same model, prefix and normalisation,
    # different destination.
    embs = encode_queries([d["query"] for d in dev])
    assert embs.shape == (n, 768), embs.shape
    nrm = np.linalg.norm(embs, axis=1)
    print(f"  embedding norms: min {nrm.min():.4f} max {nrm.max():.4f} "
          f"(should be ~1.0 if L2-normalised)")

    idx = S12.load_indices()
    print("Retrieving contexts...")
    ctx = []
    for i in range(n):
        ctx.append([S12.retrieve_context(embs[i], s, idx, S12.TOP_K) for s in SOURCES])
        if (i + 1) % 200 == 0:
            print(f"  retrieved {i+1}/{n}", flush=True)

    del idx
    S12._pos_cache.clear()
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("  released FAISS indices before loading the reader")

    llm_tok, llm = S12.load_llm()

    scores = np.zeros((n, 3))
    t0 = time.time()
    for i in range(n):
        q = dev[i]["query"]
        for j in range(3):
            scores[i, j] = S12.score_source(q, ctx[i][j], llm_tok, llm)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-9)
            print(f"  scored {i+1}/{n}  {rate:.2f} q/s  "
                  f"eta {(n-i-1)/max(rate,1e-9)/60:.1f} min", flush=True)

    # The dev retrieval-hit matrix is CPU-only work (retrieved ids vs relevance
    # annotations) and is built separately, so this GPU run stays single-purpose.
    np.savez(OUT, scores=scores, gold=gold, embs=embs)

    ok = (scores > -1e8).all(1)
    pred = scores[ok].argmax(1)
    g = gold[ok]
    m = float(np.mean([(pred[g == j] == j).mean() for j in range(3) if (g == j).sum()]))
    print(f"\nscored cleanly {ok.sum()}/{n}")
    print(f"dev likelihood-argmax macro {m:.4f}   (test value 0.5883)")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
