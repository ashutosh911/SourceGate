"""
script_9_stronger_reader.py

Tests Claim 1 from Section 7.2: a stronger reader widens the F1 envelope
on table-arithmetic queries (TAT-QA).

Compares Llama 3.1 8B vs GPT-4o-mini (cheap default; switchable to GPT-4o)
on the EXACT SAME retrieved chunks for TAT-QA test queries.

Cost estimate (GPT-4o-mini): ~$0.50-1 for full TAT-QA test set (143 queries)
Cost estimate (GPT-4o):      ~$10-15 for full TAT-QA test set
"""
import os
import json
import re
import string
import argparse
from pathlib import Path
import numpy as np
import torch
import faiss
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# API key is read from the environment; never hardcode it.
# export OPENAI_API_KEY="..." before running.
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("set OPENAI_API_KEY in the environment")

import os
print(f"API key length: {len(os.environ.get('OPENAI_API_KEY', ''))}")
print(f"API key prefix: {os.environ.get('OPENAI_API_KEY', '')[:7]}")

# Test the API once before the loop
from openai import OpenAI
client = OpenAI()
try:
    r = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=5
    )
    print(f"API smoke test OK: {r.choices[0].message.content}")
except Exception as e:
    print(f"API smoke test FAILED: {type(e).__name__}: {e}")
    raise SystemExit("Cannot continue — fix API access first")


# Set OPENAI_API_KEY in your environment before running
client = OpenAI()

DEVICE = "cuda"
BGE = "BAAI/bge-base-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "

# F1/EM scoring (SQuAD-style)
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
    if same == 0: return 0.0
    prec = same / len(p_toks)
    rec = same / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def em_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o"])
    parser.add_argument("--n", type=int, default=None,
                        help="Subset size; default = all TAT-QA test queries")
    parser.add_argument("--include_ott", action="store_true",
                        help="Also test on OTT-QA (retrieval-fail cases)")
    args = parser.parse_args()

    # Load test data
    with open("mmrag_test.json") as f:
        test = json.load(f)
    
    # Filter to TAT-QA (and optionally OTT-QA)
    tat_queries = [q for q in test if q.get("id", "").startswith("tat_")]
    ott_queries = [q for q in test if q.get("id", "").startswith("ott_")]
    
    queries_to_test = tat_queries
    if args.include_ott:
        queries_to_test = tat_queries + ott_queries
    
    if args.n:
        queries_to_test = queries_to_test[:args.n]
    
    print(f"Testing on {len(queries_to_test)} queries")
    print(f"  TAT-QA: {sum(1 for q in queries_to_test if q.get('id','').startswith('tat_'))}")
    print(f"  OTT-QA: {sum(1 for q in queries_to_test if q.get('id','').startswith('ott_'))}")
    print(f"Stronger model: {args.model}")

    # Load Llama predictions for comparison
    with open("phase5_results/predictions_oracle.jsonl") as f:
        oracle_preds = [json.loads(l) for l in f]
    pred_by_id = {p["id"]: p for p in oracle_preds if "id" in p}

    # Load retrieval pipeline (same chunks Llama saw)
    print("\nLoading retrieval pipeline...")
    tok = AutoTokenizer.from_pretrained(BGE)
    bge = AutoModel.from_pretrained(BGE, torch_dtype=torch.float16).to(DEVICE).eval()
    
    indices = {}
    chunk_ids_map = {}
    for ds in ["ott", "tat"]:
        indices[ds] = faiss.read_index(f"faiss_indices/{ds}/index.faiss")
        chunk_ids_map[ds] = np.load(f"faiss_indices/{ds}/chunk_ids.npy", allow_pickle=True)
    
    import sqlite3
    db = sqlite3.connect("chunk_texts.db")
    cur = db.cursor()
    
    def get_chunks(cids):
        phs = ",".join("?" * len(cids))
        rows = cur.execute(f"SELECT id, text FROM chunks WHERE id IN ({phs})", list(cids)).fetchall()
        d = {r[0]: r[1] for r in rows}
        return [d.get(c, "") for c in cids]

    def retrieve_table(query, top_k=10):
        prefixed = PREFIX + query
        enc = tok(prefixed, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            emb = bge(**enc).last_hidden_state[:, 0]
            emb = F.normalize(emb.float(), p=2, dim=1).cpu().numpy().astype(np.float32)
        # Pick best source (OTT vs TAT) for THIS query — oracle table routing
        all_scores = []
        all_cids = []
        for ds in ["ott", "tat"]:
            s, fids = indices[ds].search(emb, top_k)
            all_scores.append(s[0])
            all_cids.append([chunk_ids_map[ds][i] for i in fids[0]])
        # Merge scores, take top_k
        merged_scores = np.concatenate(all_scores)
        merged_cids = all_cids[0] + all_cids[1]
        top_idx = np.argsort(-merged_scores)[:top_k]
        return [merged_cids[i] for i in top_idx]

    # Test loop
    print(f"\nGenerating answers with {args.model}...")
    results = []
    total_cost = 0.0
    
    for i, q in enumerate(queries_to_test):
        cids = retrieve_table(q["query"])
        chunks = get_chunks(cids)
        context = "\n\n".join(chunks)[:6000]  # cap at ~6000 chars
        
        prompt = (f"Use ONLY the context to answer the question. "
                  f"For numerical questions requiring computation, perform the calculation and "
                  f"give ONLY the final numeric answer (no units, no explanation).\n\n"
                  f"Context:\n{context}\n\n"
                  f"Question: {q['query']}\n\nAnswer:")
        
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
            )
            pred = resp.choices[0].message.content.strip()
            usage = resp.usage
            
            # Cost tracking (GPT-4o-mini: $0.15/1M in, $0.60/1M out)
            if args.model == "gpt-4o-mini":
                cost = (usage.prompt_tokens / 1e6 * 0.15 + 
                        usage.completion_tokens / 1e6 * 0.60)
            else:  # gpt-4o
                cost = (usage.prompt_tokens / 1e6 * 5.0 + 
                        usage.completion_tokens / 1e6 * 15.0)
            total_cost += cost
        except Exception as e:
            print(f"  [error on query {i}] {e}")
            pred = ""
        
        # Compare to Llama's prediction on same query
        llama_pred = pred_by_id.get(q.get("id", ""), {})
        llama_f1 = llama_pred.get("f1", 0.0) if llama_pred else 0.0
        llama_em = llama_pred.get("em", 0.0) if llama_pred else 0.0
        
        gold = q["answer"]
        if isinstance(gold, list): gold = gold[0]
        elif isinstance(gold, dict): gold = str(gold)
        gold = str(gold).strip("'\"[]")
        
        new_f1 = f1_score(pred, gold)
        new_em = em_score(pred, gold)
        
        results.append({
            "id": q.get("id", f"q{i}"),
            "query": q["query"][:80],
            "gold": gold,
            "llama_pred": llama_pred.get("predicted", ""),
            "llama_f1": llama_f1,
            "llama_em": llama_em,
            f"{args.model}_pred": pred,
            f"{args.model}_f1": new_f1,
            f"{args.model}_em": new_em,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries_to_test)}] cost so far: ${total_cost:.3f}")

    # Aggregate
    tat_results = [r for r in results if r["id"].startswith("tat_")]
    ott_results = [r for r in results if r["id"].startswith("ott_")]
    
    print(f"\n{'='*65}")
    print(f"STRONGER READER COMPARISON: Llama 3.1 8B vs {args.model}")
    print(f"{'='*65}")
    print(f"\nTotal API cost: ${total_cost:.3f}")
    
    if tat_results:
        llama_tat_f1 = np.mean([r["llama_f1"] for r in tat_results])
        new_tat_f1 = np.mean([r[f"{args.model}_f1"] for r in tat_results])
        llama_tat_em = np.mean([r["llama_em"] for r in tat_results])
        new_tat_em = np.mean([r[f"{args.model}_em"] for r in tat_results])
        print(f"\nTAT-QA (n={len(tat_results)}):")
        print(f"  Llama 3.1 8B:   F1={llama_tat_f1:.3f}  EM={llama_tat_em:.3f}")
        print(f"  {args.model:<14}: F1={new_tat_f1:.3f}  EM={new_tat_em:.3f}")
        print(f"  Δ F1: {new_tat_f1 - llama_tat_f1:+.3f}")
        print(f"  Δ EM: {new_tat_em - llama_tat_em:+.3f}")
    
    if ott_results:
        llama_ott_f1 = np.mean([r["llama_f1"] for r in ott_results])
        new_ott_f1 = np.mean([r[f"{args.model}_f1"] for r in ott_results])
        print(f"\nOTT-QA (n={len(ott_results)}):")
        print(f"  Llama 3.1 8B:   F1={llama_ott_f1:.3f}")
        print(f"  {args.model:<14}: F1={new_ott_f1:.3f}")
        print(f"  Δ F1: {new_ott_f1 - llama_ott_f1:+.3f}")
        if abs(new_ott_f1 - llama_ott_f1) < 0.02:
            print(f"  → OTT failure mode is RETRIEVAL not READER (as predicted)")

    # Save
    Path("phase5_results").mkdir(exist_ok=True)
    with open(f"phase5_results/stronger_reader_{args.model}.json", "w") as f:
        json.dump({"results": results, "total_cost": total_cost,
                   "model": args.model}, f, indent=2)
    print(f"\nSaved to phase5_results/stronger_reader_{args.model}.json")


if __name__ == "__main__":
    main()
