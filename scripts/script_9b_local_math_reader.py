"""
script_9b_local_math_reader.py

Tests Section 7.2's claim that a stronger reader widens the F1 envelope
on table-arithmetic queries (TAT-QA). Uses a math-specialized local model
on TAT queries only, keeping Llama 3.1 8B for everything else.

This validates the paper's "routing + reader specialization" framing:
  • SourceFormer routes TAT queries to the table source (recall@picked = 0.91)
  • Llama 8B fails on arithmetic (F1 = 0.114)
  • Qwen2.5-Math-7B with the SAME retrieved chunks should compute correctly
  • If F1 jumps from 0.114 to ~0.30+, paper claim is empirically validated

Run on TAT-QA test queries only (n=143).
Compute: ~15-20 minutes on RTX 5070 Ti (4-bit Qwen-Math-7B fits in ~5 GB).
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # need to download Qwen first time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import string
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import faiss
import sqlite3
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

DEVICE = "cuda"
BGE = "BAAI/bge-base-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
QWEN_MATH = "Qwen/Qwen2.5-Math-7B-Instruct"
TOP_K = 10


# F1/EM scoring (matches script_7_evaluation.py)
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def f1_score(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p_toks)
    rec = same / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def em_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def main():
    # =====================================================================
    # 1. Load TAT-QA test queries
    # =====================================================================
    print("Loading test queries...")
    with open("mmrag_test.json") as f:
        test = json.load(f)
    tat_queries = [q for q in test if q.get("id", "").startswith("tat_")]
    print(f"  TAT-QA test queries: {len(tat_queries)}")

    # =====================================================================
    # 2. Load Llama predictions for direct comparison (oracle routing)
    # =====================================================================
    pred_files = [
        "phase5_results/predictions_oracle.jsonl",
        "phase5_results/predictions_phase4.jsonl",
        "phase5_results/predictions_phase4_seed42.jsonl",
    ]
    llama_preds_by_id = {}
    for pf in pred_files:
        if Path(pf).exists():
            with open(pf) as f:
                for line in f:
                    p = json.loads(line)
                    if p.get("id", "").startswith("tat_"):
                        # Prefer oracle predictions if available
                        if p["id"] not in llama_preds_by_id:
                            llama_preds_by_id[p["id"]] = p
            break  # use first available
    print(f"  Llama TAT predictions found: {len(llama_preds_by_id)}")

    # =====================================================================
    # 3. Load BGE encoder + TAT/OTT FAISS indices
    # =====================================================================
    print("\nLoading BGE encoder...")
    bge_tok = AutoTokenizer.from_pretrained(BGE)
    bge = AutoModel.from_pretrained(BGE, torch_dtype=torch.float16).to(DEVICE).eval()

    print("Loading FAISS indices for table sources...")
    indices = {}
    chunk_ids_map = {}
    for ds in ["ott", "tat"]:
        indices[ds] = faiss.read_index(f"faiss_indices/{ds}/index.faiss")
        chunk_ids_map[ds] = np.load(f"faiss_indices/{ds}/chunk_ids.npy",
                                     allow_pickle=True)

    # SQLite chunk lookup
    db = sqlite3.connect("chunk_texts.db")
    cur = db.cursor()

    def get_chunks(cids):
        phs = ",".join("?" * len(cids))
        rows = cur.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({phs})",
            list(cids)
        ).fetchall()
        d = {r[0]: r[1] for r in rows}
        return [d.get(c, "") for c in cids]

    @torch.no_grad()
    def encode(query):
        prefixed = PREFIX + query
        enc = bge_tok(prefixed, return_tensors="pt", truncation=True,
                      max_length=512).to(DEVICE)
        emb = bge(**enc).last_hidden_state[:, 0]
        return F.normalize(emb.float(), p=2, dim=1).cpu().numpy().astype(np.float32)

    def retrieve_table(query, top_k=TOP_K):
        """Oracle table routing: merge top results from OTT and TAT, return top-k."""
        emb = encode(query)
        all_scores = []
        all_cids = []
        for ds in ["ott", "tat"]:
            s, fids = indices[ds].search(emb, top_k)
            all_scores.append(s[0])
            all_cids.extend([chunk_ids_map[ds][i] for i in fids[0]])
        merged_scores = np.concatenate(all_scores)
        top_idx = np.argsort(-merged_scores)[:top_k]
        return [all_cids[i] for i in top_idx]

    # =====================================================================
    # 4. Free BGE before loading Qwen-Math (memory budget)
    # =====================================================================
    print("\nPre-encoding all TAT queries to free BGE from VRAM...")
    contexts_by_id = {}
    for q in tqdm(tat_queries, desc="encode+retrieve"):
        cids = retrieve_table(q["query"])
        chunks = get_chunks(cids)
        # Truncate context for math model (keep most relevant chunks first)
        ctx = "\n\n".join(chunks)
        if len(ctx) > 6000:
            ctx = ctx[:6000]
        contexts_by_id[q["id"]] = ctx

    del bge, bge_tok
    torch.cuda.empty_cache()
    print(f"  Pre-encoded {len(contexts_by_id)} TAT contexts")

    # =====================================================================
    # 5. Load Qwen2.5-Math-7B-Instruct in 4-bit
    # =====================================================================
    print(f"\nLoading {QWEN_MATH} (this may download ~7 GB on first run)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_MATH)
    if qwen_tok.pad_token is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MATH, quantization_config=bnb, device_map={"": 0},
    )
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False
    print("  Qwen-Math loaded")

    # =====================================================================
    # 6. Generate Qwen-Math predictions on each TAT query
    # =====================================================================
    print("\nGenerating Qwen-Math predictions on TAT-QA...")
    SYSTEM = (
        "You are a financial analyst. Read the context carefully, perform any "
        "required arithmetic step-by-step, then output ONLY the final numeric "
        "answer on the last line. Do not include units, currency symbols, or "
        "explanations on the final line."
    )

    def generate_qwen(query, context):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Context:\n{context}\n\nQuestion: {query}\n\n"
                f"Think step by step, then write only the final numeric "
                f"answer on the last line."},
        ]
        prompt = qwen_tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = qwen_tok(prompt, return_tensors="pt", truncation=True,
                       max_length=2500).to(DEVICE)
        with torch.no_grad():
            out = qwen.generate(
                **enc, max_new_tokens=256, do_sample=False,
                pad_token_id=qwen_tok.eos_token_id,
                temperature=1.0, top_p=1.0,
            )
        gen = qwen_tok.decode(out[0, enc["input_ids"].size(1):],
                               skip_special_tokens=True).strip()
        # Extract last line as answer
        lines = [l.strip() for l in gen.split('\n') if l.strip()]
        last = lines[-1] if lines else gen
        # Strip common decorations
        last = re.sub(r'^(answer|final answer|the answer is)[:\s]+', '',
                      last, flags=re.IGNORECASE)
        last = last.rstrip('.').strip()
        return last, gen

    results = []
    for q in tqdm(tat_queries, desc="qwen-math"):
        ctx = contexts_by_id[q["id"]]
        try:
            pred_qwen, full_gen = generate_qwen(q["query"], ctx)
        except Exception as e:
            print(f"\n  [error on {q['id']}] {e}")
            pred_qwen = ""
            full_gen = ""

        # Gold answer
        gold = q["answer"]
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        gold = str(gold).strip("'\"[]")

        # Llama prediction (if available)
        llama_p = llama_preds_by_id.get(q["id"], {})
        llama_pred = llama_p.get("predicted", "")
        llama_f1 = llama_p.get("f1", None)
        llama_em = llama_p.get("em", None)
        # If we don't have cached Llama F1, compute it from cached pred
        if llama_f1 is None and llama_pred:
            llama_f1 = f1_score(llama_pred, gold)
            llama_em = em_score(llama_pred, gold)

        qwen_f1 = f1_score(pred_qwen, gold)
        qwen_em = em_score(pred_qwen, gold)

        results.append({
            "id": q["id"],
            "query": q["query"][:100],
            "gold": gold,
            "llama_pred": llama_pred,
            "llama_f1": llama_f1,
            "llama_em": llama_em,
            "qwen_math_pred": pred_qwen,
            "qwen_math_f1": qwen_f1,
            "qwen_math_em": qwen_em,
            "qwen_math_full_gen": full_gen[-300:],  # keep end of reasoning for inspection
        })

    # =====================================================================
    # 7. Aggregate and report
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"QWEN2.5-MATH-7B vs LLAMA 3.1 8B ON TAT-QA (n={len(results)})")
    print(f"{'='*70}")

    qwen_f1s = [r["qwen_math_f1"] for r in results]
    qwen_ems = [r["qwen_math_em"] for r in results]
    llama_f1s = [r["llama_f1"] for r in results if r["llama_f1"] is not None]
    llama_ems = [r["llama_em"] for r in results if r["llama_em"] is not None]

    print(f"\n  Llama 3.1 8B (oracle routing):")
    print(f"    F1 = {np.mean(llama_f1s):.4f}  (n={len(llama_f1s)})")
    print(f"    EM = {np.mean(llama_ems):.4f}")

    print(f"\n  Qwen2.5-Math-7B (oracle routing, same chunks):")
    print(f"    F1 = {np.mean(qwen_f1s):.4f}  (n={len(qwen_f1s)})")
    print(f"    EM = {np.mean(qwen_ems):.4f}")

    print(f"\n  Δ F1: {np.mean(qwen_f1s) - np.mean(llama_f1s):+.4f}")
    print(f"  Δ EM: {np.mean(qwen_ems) - np.mean(llama_ems):+.4f}")

    # Show 5 examples
    print(f"\n{'='*70}")
    print(f"SAMPLE PREDICTIONS (first 5 where Qwen got non-zero F1)")
    print(f"{'='*70}")
    nz = [r for r in results if r["qwen_math_f1"] > 0]
    for r in nz[:5]:
        print(f"\n  Q: {r['query'][:80]}")
        print(f"  Gold:  {r['gold']!r}")
        print(f"  Llama: {r['llama_pred']!r}  F1={r['llama_f1']:.2f}")
        print(f"  Qwen:  {r['qwen_math_pred']!r}  F1={r['qwen_math_f1']:.2f}")

    # Save
    Path("phase5_results").mkdir(exist_ok=True)
    with open("phase5_results/qwen_math_tat_results.json", "w") as f:
        json.dump({
            "model": QWEN_MATH,
            "n": len(results),
            "qwen_math_f1": float(np.mean(qwen_f1s)),
            "qwen_math_em": float(np.mean(qwen_ems)),
            "llama_f1": float(np.mean(llama_f1s)) if llama_f1s else None,
            "llama_em": float(np.mean(llama_ems)) if llama_ems else None,
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved to phase5_results/qwen_math_tat_results.json")


if __name__ == "__main__":
    main()
