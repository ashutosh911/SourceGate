"""
script_34_rebuild_e5_kg_index.py  —  rebuild the E5 knowledge-graph index

THE BUG
  script_17_encoder_robustness.py line 69:

      DS_PREFIX = {..., "kg": "g.%"}

  Freebase machine ids in chunk_texts.db come in two forms. Counting rows:

      id LIKE 'g.%'      4,094
      id LIKE 'm.%'  1,223,020      <-- never selected
                     ---------
      total          1,227,114      == the BGE kg index size

  So faiss_indices_e5/kg was built from 0.33% of the knowledge graph. The
  other four E5 indices match BGE exactly; only kg is affected. The
  "already built" guard in build_e5_index only checks that the file exceeds
  1024 bytes, and the truncated index is 12 MB, so it was silently skipped on
  every later run.

  Consequence: every E5 number in Section 6.10 is invalid, including the
  recall@picked figure the manuscript currently presents as the one
  encoder-comparable finding. KG-labelled queries (210 of 1,286) could not
  retrieve their evidence at all.

WHAT THIS DOES
  Rebuilds faiss_indices_e5/kg over BOTH id families, mirroring
  build_e5_index() exactly otherwise: same E5 model, "passage: " prefix,
  attention-masked mean pooling (NOT CLS), L2 normalisation, IndexFlatIP,
  ORDER BY id.

  The old index is moved aside rather than deleted, so the broken artifact
  remains available for inspection.

USAGE
  conda activate chestx && python script_34_rebuild_e5_kg_index.py

OUTPUT
  faiss_indices_e5/kg/{index.faiss,chunk_ids.npy}   (rebuilt)
  faiss_indices_e5/kg/broken_g_only/                (previous, preserved)
"""

import gc
import shutil
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

E5_MODEL = "intfloat/e5-base-v2"
E5_DIM = 768
E5_PASSAGE_PREFIX = "passage: "
CHUNK_DB = "chunk_texts.db"
OUT = Path("faiss_indices_e5/kg")
BATCH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Both Freebase id families. This is the fix.
KG_WHERE = "id LIKE 'm.%' OR id LIKE 'g.%'"


def e5_pool(h, mask):
    """E5 requires attention-masked MEAN pooling (not BGE's CLS token)."""
    masked = h.masked_fill(~mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)


def main():
    conn = sqlite3.connect(CHUNK_DB)
    total = conn.execute(f"SELECT COUNT(*) FROM chunks WHERE {KG_WHERE}").fetchone()[0]
    print(f"KG chunks to index: {total:,}")
    ref = faiss.read_index("faiss_indices/kg/index.faiss").ntotal
    print(f"BGE kg index size : {ref:,}")
    if total != ref:
        print(f"  WARNING: selector yields {total:,}, BGE has {ref:,}")

    if (OUT / "index.faiss").exists():
        bak = OUT / "broken_g_only"
        bak.mkdir(exist_ok=True)
        for f in ("index.faiss", "chunk_ids.npy"):
            if (OUT / f).exists():
                shutil.move(str(OUT / f), str(bak / f))
        print(f"  previous (truncated) index moved to {bak}")

    tok = AutoTokenizer.from_pretrained(E5_MODEL)
    model = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(DEVICE).eval()

    index = faiss.IndexFlatIP(E5_DIM)
    all_ids, bi, bt = [], [], []

    @torch.no_grad()
    def flush():
        if not bi:
            return
        enc = tok([E5_PASSAGE_PREFIX + (t or "") for t in bt],
                  padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
        emb = e5_pool(model(**enc).last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=1)
        index.add(emb.cpu().numpy().astype("float32"))
        all_ids.extend(bi)

    t0 = time.time()
    cur = conn.execute(f"SELECT id, text FROM chunks WHERE {KG_WHERE} ORDER BY id")
    for n, (cid, text) in enumerate(cur, 1):
        bi.append(cid)
        bt.append(text)
        if len(bi) >= BATCH:
            flush()
            bi, bt = [], []
            if index.ntotal % (BATCH * 100) == 0:
                el = time.time() - t0
                print(f"  {index.ntotal:>9,}/{total:,}  "
                      f"{index.ntotal/max(el,1):>7.0f}/s  "
                      f"eta {(total-index.ntotal)/max(index.ntotal/max(el,1),1)/60:>5.1f} min")
    flush()
    conn.close()

    assert index.ntotal == len(all_ids) == total, \
        f"count mismatch: index {index.ntotal}, ids {len(all_ids)}, expected {total}"

    faiss.write_index(index, str(OUT / "index.faiss"))
    np.save(OUT / "chunk_ids.npy", np.array(all_ids, dtype=object))
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    el = time.time() - t0
    print(f"\nRebuilt: {index.ntotal:,} vectors in {el/60:.1f} min "
          f"({index.ntotal/max(el,1):.0f}/s)")
    print(f"  index.faiss  {(OUT/'index.faiss').stat().st_size/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
