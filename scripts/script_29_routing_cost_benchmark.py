"""
script_29_routing_cost_benchmark.py  —  Measured per-query routing overhead

PURPOSE
  Table 10 reports the per-query cost of each routing strategy as *counts*
  (FAISS searches, reader forward passes, MLP passes). Reviewer 2 asked for
  query latency, cost per query and memory against the configuration that does
  not route, on the grounds that counts alone do not establish whether routing
  is worthwhile. This script measures the wall-clock and memory figures that
  turn those counts into costs.

SCOPE
  We time the ROUTING OVERHEAD only: everything a method does to decide which
  source to search, plus the retrieval it then performs. Answer generation is
  deliberately excluded because it is identical across routing strategies (the
  same reader, the same 10-chunk context budget) and would swamp the
  differences the table is about. The one exception is PrefRAG-Conf, whose
  reader forward passes ARE the routing decision and are therefore timed.

  All timings are single-query (batch size 1), which is the serving regime the
  comparison is about.

MEASURED COMPONENTS
  encode      BGE-base query encoding (shared by every method)
  mlp         SourceGate 540K-parameter forward pass
  faiss_<ds>  top-10 exact IndexFlatIP search against one dataset index
  reader      one frozen-reader forward pass scoring a retrieved context

MEMORY
  Resident index memory is reported per dataset index. Note that routing does
  not reduce it: the router cannot know which index it will need until it has
  run, so all K indices must stay resident either way. What routing reduces is
  search work, not index footprint -- we state this explicitly rather than
  claim a memory saving the method does not deliver.

USAGE
  conda activate chestx && python script_29_routing_cost_benchmark.py \
      [--n 100] [--device cpu|gpu] [--skip-reader]

OUTPUT
  phase5_results/routing_cost_benchmark.json
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

import faiss
import torch

RESULTS_DIR = Path("phase5_results")
FAISS_DIR = Path("faiss_indices")
TEST_FILE = "mmrag_test.json"
BGE_NAME = "BAAI/bge-base-en-v1.5"
LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TOP_K = 10

DATASETS = ["nq", "triviaqa", "ott", "tat", "kg"]
TYPE_TO_DATASETS = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
SOURCE_TYPES = ["text", "table", "kg"]

# Per-query component counts, mirroring Table 10.
# (n_faiss_by_type is resolved per query from the actual routing decision.)
STRATEGIES = {
    # Random/Majority still encode the query: the routing decision needs no
    # embedding, but the retrieval that follows it does, so every strategy
    # pays the encoder cost exactly once.
    "Random / Majority":      dict(encode=True,  mlp=0, faiss="one_type", reader=0),
    "SourceGate (Superv.)":   dict(encode=True,  mlp=1, faiss="one_type", reader=0),
    "PrefRAG-Conf":           dict(encode=True,  mlp=0, faiss="all_types", reader=3),
    "No-routing (union)":     dict(encode=True,  mlp=0, faiss="all_five", reader=0),
    "Hybrid (tau=0.70)":      dict(encode=True,  mlp=1, faiss="hybrid", reader=0),
}
HYBRID_ROUTE_FRACTION = 0.544   # Section 6.9


def timeit(fn, n_warmup, n_rep, sync=False):
    """Return (mean_ms, std_ms) over n_rep timed calls after n_warmup warmups."""
    for _ in range(n_warmup):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="queries to time over")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="FAISS search device")
    ap.add_argument("--skip-reader", action="store_true",
                    help="skip the frozen-reader forward-pass timing")
    ap.add_argument("--out", default=str(RESULTS_DIR / "routing_cost_benchmark.json"))
    args = ap.parse_args()

    out = {"config": {"n_queries": args.n, "faiss_device": args.device,
                      "top_k": TOP_K, "batch_size": 1}}

    print("Loading test data and query embeddings...")
    test_data = json.load(open(TEST_FILE))
    cache = Path("query_emb_cache/test_embs_k3.npy")
    if not cache.exists():
        cache = Path("query_emb_cache/test_embs.npy")
    embs = np.load(cache).astype(np.float32)
    n = min(args.n, len(embs))
    print(f"  timing over {n} queries")

    # ── index memory ──────────────────────────────────────────────────────────
    print("\nIndex footprint...")
    mem = {}
    for ds in DATASETS:
        p = FAISS_DIR / ds / "index.faiss"
        mem[ds] = {"bytes_on_disk": p.stat().st_size,
                   "gib_on_disk": p.stat().st_size / 1024 ** 3}
        print(f"  {ds:<9} {mem[ds]['gib_on_disk']:.2f} GiB")
    total_gib = sum(v["gib_on_disk"] for v in mem.values())
    out["index_memory"] = {
        "per_dataset": mem,
        "total_gib": total_gib,
        "note": ("All five indices must remain resident under every strategy, "
                 "including routing: the router cannot know which index it "
                 "needs before it runs. Routing reduces search work, not "
                 "index footprint."),
    }
    print(f"  {'TOTAL':<9} {total_gib:.2f} GiB (resident under every strategy)")

    # ── FAISS search ──────────────────────────────────────────────────────────
    print(f"\nLoading FAISS indices ({args.device})...")
    indices, gpu_res = {}, None
    if args.device == "gpu":
        gpu_res = faiss.StandardGpuResources()
    for ds in DATASETS:
        idx = faiss.read_index(str(FAISS_DIR / ds / "index.faiss"))
        if args.device == "gpu":
            idx = faiss.index_cpu_to_gpu(gpu_res, 0, idx)
        indices[ds] = idx
        print(f"  {ds}: {idx.ntotal} vectors")

    print("\nTiming FAISS top-10 search (batch size 1)...")
    faiss_ms = {}
    for ds in DATASETS:
        qs = [embs[i % n].reshape(1, -1) for i in range(n)]
        counter = {"i": 0}

        def search2(_ds=ds, _qs=qs, _c=counter):
            q = _qs[_c["i"] % n]
            _c["i"] += 1
            indices[_ds].search(q, TOP_K)

        m, s = timeit(search2, n_warmup=5, n_rep=n)
        faiss_ms[ds] = {"mean_ms": m, "std_ms": s}
        print(f"  {ds:<9} {m:8.3f} ± {s:.3f} ms")
    out["faiss_search_ms"] = faiss_ms

    # ── BGE query encoding ────────────────────────────────────────────────────
    print("\nTiming BGE query encoding (batch size 1)...")
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    btok = AutoTokenizer.from_pretrained(BGE_NAME)
    bge = AutoModel.from_pretrained(BGE_NAME, dtype=torch.float16).to(dev).eval()
    queries = [test_data[i]["query"] for i in range(n)]
    prefix = "Represent this sentence for searching relevant passages: "
    c = {"i": 0}

    def encode():
        q = prefix + queries[c["i"] % n]
        c["i"] += 1
        enc = btok(q, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            bge(**enc)

    m, s = timeit(encode, n_warmup=5, n_rep=n, sync=True)
    out["bge_encode_ms"] = {"mean_ms": m, "std_ms": s}
    print(f"  encode    {m:8.3f} ± {s:.3f} ms")
    del bge
    torch.cuda.empty_cache()

    # ── SourceGate MLP ────────────────────────────────────────────────────────
    print("\nTiming SourceGate MLP forward (batch size 1)...")
    from sourceformer import SourceFormerK3
    sg = SourceFormerK3(dropout=0.2).to(dev).eval()
    n_params = sum(p.numel() for p in sg.parameters())
    qt = torch.from_numpy(embs[:n]).float().to(dev)
    c2 = {"i": 0}

    def mlp():
        with torch.no_grad():
            sg(qt[c2["i"] % n].unsqueeze(0))
        c2["i"] += 1

    m, s = timeit(mlp, n_warmup=10, n_rep=n, sync=True)
    out["sourcegate_mlp_ms"] = {"mean_ms": m, "std_ms": s, "n_params": int(n_params)}
    print(f"  mlp       {m:8.3f} ± {s:.3f} ms   ({n_params:,} params)")
    del sg
    torch.cuda.empty_cache()

    # ── frozen reader forward pass ────────────────────────────────────────────
    if args.skip_reader:
        print("\nSkipping reader timing (--skip-reader).")
        out["reader_forward_ms"] = None
    else:
        print(f"\nLoading {LLM_NAME} (4-bit NF4) for forward-pass timing...")
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        torch.cuda.reset_peak_memory_stats(0)
        ltok = AutoTokenizer.from_pretrained(LLM_NAME)
        if ltok.pad_token is None:
            ltok.pad_token = ltok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, quantization_config=bnb, device_map={"": 0},
            low_cpu_mem_usage=True)
        llm.eval()
        load_gib = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"  reader resident VRAM: {load_gib:.2f} GiB")

        # A representative PrefRAG-Conf scoring input: ~250-token context
        # (the text/table mean reported in Section 6.2) plus the question.
        ctx = ("The document describes the subject in detail. " * 40)
        prompt = ("Answer the following question using only the provided "
                  "context. Be concise.\n\nContext:\n" + ctx +
                  "\n\nQuestion: " + queries[0] + "\n\nAnswer:")
        enc = ltok(prompt, return_tensors="pt", truncation=True,
                   max_length=1024).to("cuda")
        labels = enc["input_ids"].clone()
        seq_len = int(enc["input_ids"].shape[1])

        def reader():
            with torch.no_grad():
                llm(**enc, labels=labels)

        n_rep = max(10, n // 5)
        m, s = timeit(reader, n_warmup=3, n_rep=n_rep, sync=True)
        peak_gib = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        out["reader_forward_ms"] = {"mean_ms": m, "std_ms": s, "n_rep": n_rep,
                                    "seq_len": seq_len,
                                    "resident_vram_gib": load_gib,
                                    "peak_vram_gib": peak_gib}
        print(f"  reader    {m:8.3f} ± {s:.3f} ms  (seq_len={seq_len}, "
              f"peak VRAM {peak_gib:.2f} GiB)")
        del llm
        torch.cuda.empty_cache()

    # ── assemble per-strategy per-query latency ───────────────────────────────
    print("\nPer-query routing overhead by strategy:")
    enc_ms = out["bge_encode_ms"]["mean_ms"]
    mlp_ms = out["sourcegate_mlp_ms"]["mean_ms"]
    rd_ms = (out["reader_forward_ms"] or {}).get("mean_ms")

    # mean FAISS cost of searching one source TYPE, weighted by how often
    # SourceGate actually routes there (text = nq+triviaqa, table = ott+tat)
    type_cost = {t: sum(faiss_ms[d]["mean_ms"] for d in ds)
                 for t, ds in TYPE_TO_DATASETS.items()}
    all_five = sum(faiss_ms[d]["mean_ms"] for d in DATASETS)
    all_types = sum(type_cost.values())   # == all_five, kept explicit
    # routing mix from the supervised-pretraining router on the test set
    sg_ckpt = Path("checkpoints/sourceformer_k3_seed42_best.pt")
    if sg_ckpt.exists():
        from sourceformer import SourceFormerK3 as SG2
        ck = torch.load(sg_ckpt, map_location="cpu", weights_only=False)
        m2 = SG2(dropout=0.2)
        m2.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m2.eval()
        with torch.no_grad():
            dec = m2(torch.from_numpy(embs).float()).argmax(-1).numpy()
        mix = {t: float((dec == i).mean()) for i, t in enumerate(SOURCE_TYPES)}
    else:
        mix = {t: 1 / 3 for t in SOURCE_TYPES}
    one_type = sum(mix[t] * type_cost[t] for t in SOURCE_TYPES)
    out["routing_mix"] = mix
    out["faiss_cost_ms"] = {"one_type_expected": one_type,
                            "per_type": type_cost, "all_five": all_five}

    rows = {}
    for name, spec in STRATEGIES.items():
        f = spec["faiss"]
        faiss_cost = {"one_type": one_type, "all_types": all_types,
                      "all_five": all_five,
                      "hybrid": HYBRID_ROUTE_FRACTION * one_type +
                                (1 - HYBRID_ROUTE_FRACTION) * all_five}[f]
        total = faiss_cost
        total += enc_ms if spec["encode"] else 0.0
        total += mlp_ms * spec["mlp"]
        reader_cost = None
        if spec["reader"]:
            if rd_ms is None:
                total = None
            else:
                reader_cost = rd_ms * spec["reader"]
                total += reader_cost
        rows[name] = {"encode_ms": enc_ms if spec["encode"] else 0.0,
                      "mlp_ms": mlp_ms * spec["mlp"],
                      "faiss_ms": faiss_cost,
                      "reader_ms": reader_cost,
                      "total_ms": total}
        t = "n/a" if total is None else f"{total:8.2f}"
        print(f"  {name:<22} {t} ms   "
              f"(faiss {faiss_cost:.2f}"
              + (f", reader {reader_cost:.2f}" if reader_cost else "") + ")")
    out["per_strategy_ms"] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
