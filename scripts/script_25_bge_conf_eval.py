"""
script_25_bge_conf_eval.py — BGE-confidence end-to-end F1/EM evaluation

Imports retrieve_context directly from script_12_prefrag_conf.py to guarantee
identical retrieval behaviour to the existing evaluation pipeline.

Readers:
  --reader openai    GPT-4o-mini (matches script_10's "openai" config)
  --reader llama70b  Llama-3.3-70B-Instruct-Turbo via Together API
                      (matches script_10's "together" config exactly —
                      use THIS for any comparison against SG-Phase3 / Oracle /
                      Random / etc. "Llama F1" numbers in Table 9, since those
                      were all generated with the 70B model, not the 8B one)
  --reader llama     Llama-3.1-8B-Instruct, local 4-bit — kept for reference /
                      cheap smoke-testing only. NOT model-comparable to the
                      70B numbers already in Table 9. Do not report this
                      column as "Llama F1" against SG-Phase3 etc. without a
                      clear footnote that it is a different, smaller model.

USAGE:
  export OPENAI_API_KEY="sk-..."
  export TOGETHER_API_KEY="..."
  python script_25_bge_conf_eval.py --reader openai
  python script_25_bge_conf_eval.py --reader llama70b
  python script_25_bge_conf_eval.py --reader llama70b --n 20   # smoke test
"""

import argparse, json, os, re, string, time
from pathlib import Path
from collections import Counter

import numpy as np
import requests
import torch

# Import only load_indices and retrieve_chunk_ids from script_12
from script_12_prefrag_conf import (
    load_indices, retrieve_chunk_ids,
    TYPE_TO_DATASETS_K3, SOURCE_TYPES_K3, SOURCE_IDX,
    TOP_K,
)

import sqlite3 as _sqlite3
import faiss as _faiss

CHUNK_DB          = Path("chunk_texts.db")
FAISS_DIR_G       = Path("faiss_indices")
GEN_MAX_CTX_CHARS = 3000

# Build positional cache using chunk_ids.npy order (matches FAISS build order)
# NOT SQLite ORDER BY id (which is lexicographic and doesn't match)
_pos_cache_gen = {}   # ds -> list[text] in FAISS positional order

def _build_pos_cache_gen(ds):
    """Build positional text cache using chunk_ids.npy order."""
    if ds in _pos_cache_gen:
        return
    # Step 1: load ordered chunk IDs from FAISS build artefact
    id_path = FAISS_DIR_G / ds / "chunk_ids.npy"
    if not id_path.exists():
        _pos_cache_gen[ds] = []
        return
    ordered_ids = list(np.load(id_path, allow_pickle=True))

    # Step 2: fetch text for each ID from SQLite in batches (SQLite limit ~999 vars)
    conn = _sqlite3.connect(str(CHUNK_DB))
    id_to_text = {}
    BATCH = 500
    for start in range(0, len(ordered_ids), BATCH):
        batch = ordered_ids[start:start + BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            batch).fetchall()
        id_to_text.update({r[0]: r[1] for r in rows})
    conn.close()
    # Step 3: build positional list matching chunk_ids.npy order
    _pos_cache_gen[ds] = [id_to_text.get(cid, "") for cid in ordered_ids]
    hits = sum(1 for t in _pos_cache_gen[ds] if t)
    print(f"  {ds}: {hits:,}/{len(ordered_ids):,} chunks loaded (chunk_ids.npy order)")

def retrieve_context_gen(q_emb, source_type, faiss_indices, k=TOP_K):
    """Retrieve top-k chunks, merged and re-ranked by score across datasets
    within a source type — matches search_type()'s pool-then-rerank logic
    in script_10_reader_scaling.py, instead of taking top-k per dataset
    and concatenating (which let dataset list-order silently starve the
    other dataset's chunks out of the truncated context)."""
    datasets = TYPE_TO_DATASETS_K3[source_type]

    if len(datasets) == 1:
        ds = datasets[0]
        _build_pos_cache_gen(ds)
        ids = retrieve_chunk_ids(q_emb, ds, faiss_indices[ds], k)
        cache = _pos_cache_gen[ds]
        texts = [cache[i] for i in ids if 0 <= i < len(cache) and cache[i]]
        return "\n\n".join(texts)[:GEN_MAX_CTX_CHARS]

    all_scores, all_texts = [], []
    for ds in datasets:
        _build_pos_cache_gen(ds)
        scores, faiss_ids = faiss_indices[ds].search(
            q_emb[None].astype(np.float32), k)
        cache = _pos_cache_gen[ds]
        for score, fid in zip(scores[0], faiss_ids[0]):
            if 0 <= fid < len(cache) and cache[fid]:
                all_scores.append(score)
                all_texts.append(cache[fid])

    order = np.argsort(-np.array(all_scores))[:k]
    texts = [all_texts[i] for i in order]
    return "\n\n".join(texts)[:GEN_MAX_CTX_CHARS]

RESULTS_DIR    = Path("phase5_results")
TEST_FILE      = "mmrag_test.json"
DECISIONS_FILE = RESULTS_DIR / "confidence_decisions_k3.npy"
EMB_CACHE_PATH = Path("query_emb_cache") / "test_embs.npy"

MAX_NEW_TOKENS  = 32
# Matches script_10_reader_scaling.py's SYSTEM_PROMPT exactly — this is the
# prompt that was verified (1/1286 refusal-pattern predictions on SG-Phase3)
# to avoid the verbose-refusal failure mode. Do not weaken this.
SYSTEM_PROMPT   = ("Answer the question based on the provided context. Be concise. "
                    "Respond with ONLY the answer in a few words, no explanation. "
                    "If unsure, give your best guess.")

# Together API config — mirrors script_10_reader_scaling.py's
# READER_CONFIGS["together"] exactly.
TOGETHER_MODEL    = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TOGETHER_API_BASE = "https://api.together.xyz/v1/chat/completions"
TOGETHER_ENV_KEY  = "TOGETHER_API_KEY"
MAX_RETRIES        = 5
RETRY_BASE_DELAY   = 2.0

# ── SQuAD F1/EM ──────────────────────────────────────────────────────────────
def normalize_answer(s):
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t): return ' '.join(t.split())
    def remove_punc(t):
        excl = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in excl)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def get_tokens(s): return normalize_answer(s).split()

def compute_f1(pred, gold):
    p_toks = get_tokens(pred); g_toks = get_tokens(gold)
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0: return 0.0
    p = n / len(p_toks); r = n / len(g_toks)
    return 2 * p * r / (p + r)

def score(pred, gold):
    if isinstance(gold, list):
        f1 = max(compute_f1(pred, g) for g in gold)
        em = float(any(normalize_answer(pred) == normalize_answer(g) for g in gold))
    else:
        f1 = compute_f1(pred, gold)
        em = float(normalize_answer(pred) == normalize_answer(gold))
    return f1, em

# ── Prompts (local 8B path only) ──────────────────────────────────────────────
LLAMA_TMPL = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
              f"{SYSTEM_PROMPT}<|eot_id|>\n"
              "<|start_header_id|>user<|end_header_id|>\n"
              "Context: {context}\nQuestion: {query}<|eot_id|>\n"
              "<|start_header_id|>assistant<|end_header_id|>\n")

# ── Local Llama-3.1-8B generation (reference / smoke-test only) ──────────────
def load_llama():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print("Loading Llama 3.1 8B (4-bit NF4)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=bnb, device_map="auto")
    model.eval()
    return model, tok

@torch.no_grad()
def generate_llama(model, tok, query, context):
    prompt = LLAMA_TMPL.format(context=context[:3000], query=query)
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    if inp.input_ids.shape[1] > 1400: return ""
    out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ── Llama-3.3-70B via Together API (matches script_10 exactly) ───────────────
def generate_together(query, context, api_key):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # NOTE: matches script_10's exact context/question formatting
        # ("Context:\n{context}\n\nQuestion: {query}"), not script_25's
        # older single-line OpenAI formatting — kept identical to script_10
        # here since this path exists specifically for apples-to-apples
        # comparison against script_10's Llama numbers.
        {"role": "user", "content": f"Context:\n{context[:3000]}\n\nQuestion: {query}"},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                TOGETHER_API_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": TOGETHER_MODEL,
                    "messages": messages,
                    "max_tokens": MAX_NEW_TOKENS,
                    "temperature": 0,
                    "stop": ["<|eot_id|>"],
                },
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Together API {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err_str = str(e)
            if attempt < MAX_RETRIES - 1 and any(
                x in err_str.lower()
                for x in ["rate limit", "429", "503", "timeout", "server error", "500"]
            ):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Retry {attempt+1}/{MAX_RETRIES} after {delay:.1f}s: {err_str[:80]}")
                time.sleep(delay)
            else:
                print(f"  Together error: {e}")
                return ""
    return ""

# ── OpenAI generation ─────────────────────────────────────────────────────────
def generate_openai(client, query, context, model="gpt-4o-mini"):
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=MAX_NEW_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Context: {context[:3000]}\nQuestion: {query}"},
            ])
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"  OpenAI error: {e}")
        return ""

# ── main evaluation ───────────────────────────────────────────────────────────
def evaluate(reader, test_data, decisions, faiss_indices,
             test_embs, cache_path, n=None):
    n = n or len(test_data)
    test_data = test_data[:n]

    cache = {}
    if cache_path.exists():
        cache = json.load(open(cache_path))
        print(f"  Loaded {len(cache)} cached entries")

    # Pre-load all dataset pos caches using correct chunk_ids.npy ordering
    all_ds = set(ds for dss in TYPE_TO_DATASETS_K3.values() for ds in dss)
    for ds in all_ds:
        _build_pos_cache_gen(ds)

    if reader == "llama":
        model, tok = load_llama()
    elif reader == "llama70b":
        together_key = os.environ.get(TOGETHER_ENV_KEY)
        if not together_key:
            raise RuntimeError(f"Set {TOGETHER_ENV_KEY} to use --reader llama70b")
    else:
        from openai import OpenAI
        client = OpenAI()

    results = []
    t0 = time.time()

    for i, item in enumerate(test_data):
        key = str(i)
        if key in cache:
            results.append(cache[key])
            continue

        src = SOURCE_TYPES_K3[decisions[i]]

        # Use generation-appropriate context (3000 chars, not 800),
        # merged and re-ranked by score across constituent datasets
        # (matches search_type()'s pool-then-rerank logic).
        context = retrieve_context_gen(test_embs[i], src, faiss_indices)

        if reader == "llama":
            pred = generate_llama(model, tok, item["query"], context)
        elif reader == "llama70b":
            pred = generate_together(item["query"], context, together_key)
        else:
            pred = generate_openai(client, item["query"], context)

        gold_raw = item.get("answer", "")
        # mmrag_test.json stores answers as Python-list-formatted strings e.g. "['Michael Gambon']"
        # Parse them the same way script_15 does
        if isinstance(gold_raw, str) and gold_raw.startswith("["):
            try:
                import ast
                parsed = ast.literal_eval(gold_raw)
                gold = parsed if isinstance(parsed, list) else [gold_raw]
            except Exception:
                gold = [gold_raw]
        elif isinstance(gold_raw, list):
            gold = gold_raw
        else:
            gold = [gold_raw]
        f1, em = score(pred, gold)

        # Oracle type from dataset_score
        ds = item["dataset_score"]
        by_type = {
            "text":  max(ds.get("nq",0), ds.get("triviaqa",0)),
            "table": max(ds.get("ott",0), ds.get("tat",0)),
            "kg":    ds.get("kg",0),
        }
        oracle_type = max(by_type, key=by_type.get)

        r = {"id": item.get("id", str(i)), "query": item["query"],
             "gold": gold, "predicted": pred,
             "picked_type": src, "oracle_type": oracle_type,
             "f1": f1, "em": em}
        results.append(r)
        cache[key] = r

        if (i + 1) % 50 == 0:
            with open(cache_path, "w") as f: json.dump(cache, f)
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n - i - 1)
            f1_run = np.mean([x["f1"] for x in results])
            print(f"  [{i+1}/{n}] F1={f1_run:.4f}  ETA={eta/60:.0f}min")

    with open(cache_path, "w") as f: json.dump(cache, f)

    f1_mean = float(np.mean([r["f1"] for r in results]))
    em_mean = float(np.mean([r["em"] for r in results]))

    # Per oracle type breakdown
    per_type = {}
    for src in SOURCE_TYPES_K3:
        sub = [r for r in results if r["oracle_type"] == src]
        if sub:
            per_type[src] = {"n": len(sub),
                             "f1": float(np.mean([r["f1"] for r in sub])),
                             "em": float(np.mean([r["em"] for r in sub]))}

    print(f"\n  [{reader}] BGE-conf F1={f1_mean:.4f}  EM={em_mean:.4f}")
    for src, v in per_type.items():
        print(f"    {src}: F1={v['f1']:.4f}  n={v['n']}")

    return {"n": n, "reader": reader, "f1_mean": f1_mean, "em_mean": em_mean,
            "per_type": per_type}, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=["llama","llama70b","openai","both"], default="both")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    print("Loading test data...")
    with open(TEST_FILE) as f:
        test_data = json.load(f)

    decisions = np.load(DECISIONS_FILE)
    print(f"BGE-conf routing: text={sum(decisions==0)}, "
          f"table={sum(decisions==1)}, kg={sum(decisions==2)}")

    test_embs = np.load(EMB_CACHE_PATH).astype(np.float32)
    print(f"Embeddings: {test_embs.shape}")

    print("Loading FAISS indices...")
    faiss_indices = load_indices()

    summaries = {}

    if args.reader in ("llama", "both"):
        cache_path = RESULTS_DIR / "bge_conf_eval_llama_cache.json"
        s, _ = evaluate("llama", test_data, decisions, faiss_indices,
                        test_embs, cache_path, args.n)
        summaries["llama"] = s
        with open(RESULTS_DIR/"bge_conf_eval_llama.json","w") as f: json.dump(s,f,indent=2)

    if args.reader == "llama70b":
        cache_path = RESULTS_DIR / "bge_conf_eval_llama70b_cache.json"
        s, _ = evaluate("llama70b", test_data, decisions, faiss_indices,
                        test_embs, cache_path, args.n)
        summaries["llama70b"] = s
        with open(RESULTS_DIR/"bge_conf_eval_llama70b.json","w") as f: json.dump(s,f,indent=2)

    if args.reader in ("openai", "both"):
        cache_path = RESULTS_DIR / "bge_conf_eval_openai_cache.json"
        s, _ = evaluate("openai", test_data, decisions, faiss_indices,
                        test_embs, cache_path, args.n)
        summaries["openai"] = s
        with open(RESULTS_DIR/"bge_conf_eval_openai.json","w") as f: json.dump(s,f,indent=2)

    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    rows = [
        ("No-routing (union)",  0.1770, 0.4420),
        ("SG supervised",       0.1700, 0.4370),
        ("SG joint",            0.1690, 0.4350),
    ]
    print(f"  {'Method':<22} {'Llama-70B F1':>13} {'GPT-4o F1':>12}")
    print("  " + "-"*49)
    for name, lf, of in rows:
        print(f"  {name:<22} {lf:>13.4f} {of:>12.4f}")
    lf = summaries.get("llama70b", summaries.get("llama", {})).get("f1_mean", "---")
    of = summaries.get("openai", {}).get("f1_mean", "---")
    lf_s = f"{lf:.4f}" if isinstance(lf, float) else lf
    of_s = f"{of:.4f}" if isinstance(of, float) else of
    print(f"  {'BGE-confidence':<22} {lf_s:>13} {of_s:>12}")

    out = RESULTS_DIR / "bge_conf_eval_summary.json"
    with open(out, "w") as f: json.dump(summaries, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
