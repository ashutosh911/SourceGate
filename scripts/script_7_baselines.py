"""
Script 7: Phase 6 baselines — all 4 baselines evaluated on the same test set.

Baselines:
  B1: Random routing   — lower bound
  B2: Oracle routing   — upper bound (perfect router)
  B3: MLP classifier   — disjoint pipeline (no gradient from answer → router)
  B4: Majority routing — naive, always picks most common source

All baselines use the same AnswerFormer (loaded from best joint checkpoint's
LoRA weights) so the only variable is routing.

Output: baselines_results.json
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from sourceformer import SOURCES, SOURCE_TO_IDX, K, EMBED_DIM, build_sourceformer

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BGE_NAME  = "BAAI/bge-base-en-v1.5"
LLM_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "
TOP_K         = 5
MAX_QUERY_LEN = 512
MAX_CTX_LEN   = 1024
MAX_GEN_LEN   = 64
INDICES_DIR   = Path("faiss_indices")
CHUNKS_DIR    = Path("chunks_by_source")
CKPT_DIR      = Path("checkpoints")
BEST_SEED     = 42   # change to whichever seed produced best joint EM


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------

def oracle_label(item):
    scores = item["dataset_score"]
    max_s  = max(scores.values())
    if max_s == 0:
        return None
    winners = [s for s in SOURCES if scores.get(s, 0) == max_s]
    return SOURCE_TO_IDX[winners[0]]


with open("mmrag_dev.json") as f:
    raw_data = json.load(f)

test_data = []
for item in raw_data:
    label = oracle_label(item)
    if label is None:
        continue
    test_data.append({
        "query":  item["query"],
        "answer": item.get("answer", ""),
        "label":  label,
    })
print(f"Test set: {len(test_data):,} items")

# Label distribution (for majority baseline)
label_counts = Counter(d["label"] for d in test_data)
majority_src = label_counts.most_common(1)[0][0]
print(f"Majority source: {SOURCES[majority_src]} "
      f"({label_counts[majority_src]/len(test_data)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

print(f"\nLoading BGE...")
bge_tok   = AutoTokenizer.from_pretrained(BGE_NAME)
bge_model = AutoModel.from_pretrained(BGE_NAME, torch_dtype=torch.float32).to(DEVICE).eval()


@torch.inference_mode()
def encode_query(query: str) -> np.ndarray:
    enc = bge_tok(QUERY_PREFIX + query, return_tensors="pt",
                  truncation=True, max_length=MAX_QUERY_LEN).to(DEVICE)
    emb = bge_model(**enc).last_hidden_state[:, 0]
    return F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()


# Pre-encode all test queries (once, reused across all baselines)
print("Encoding test queries...")
test_embs = np.stack([encode_query(d["query"]) for d in tqdm(test_data)])
print(f"  Encoded {len(test_embs):,} queries")

print(f"\nLoading FAISS indices...")
indexes = {}
cid_maps = {}
for src in SOURCES:
    p = INDICES_DIR / src
    indexes[src]  = faiss.read_index(str(p / "index.faiss"))
    cid_maps[src] = np.load(p / "chunk_ids.npy", allow_pickle=True)
    print(f"  {src}: {indexes[src].ntotal:,} vectors")


def retrieve(src: str, q_emb: np.ndarray, k: int = TOP_K) -> list[str]:
    _, ids = indexes[src].search(q_emb.reshape(1, -1).astype(np.float32), k)
    return [cid_maps[src][i] for i in ids[0]]


chunk_stores: dict[str, dict] = {}
def get_chunk_text(src: str, cid: str) -> str:
    if src not in chunk_stores:
        with open(CHUNKS_DIR / f"{src}.jsonl") as f:
            chunk_stores[src] = {json.loads(l)["id"]: json.loads(l)["text"]
                                 for l in f}
    return chunk_stores[src].get(cid, "")


print(f"\nLoading AnswerFormer ({LLM_NAME})...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)
llm_tok = AutoTokenizer.from_pretrained(LLM_NAME)
llm_tok.pad_token     = llm_tok.eos_token
llm_tok.padding_side  = "right"

base_llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16)

# Load LoRA from best joint training run
lora_path = CKPT_DIR / f"joint_seed{BEST_SEED}_lora"
if lora_path.exists():
    llm = PeftModel.from_pretrained(base_llm, str(lora_path))
    print(f"  LoRA loaded from {lora_path}")
else:
    llm = base_llm
    print(f"  WARNING: No LoRA checkpoint found at {lora_path}. Using base model.")

llm.eval()


def build_prompt(query: str, evidence: list[str]) -> str:
    ev_text = "\n\n".join(f"[Passage {i+1}] {c}" for i, c in enumerate(evidence))
    return (f"<|system|>\nYou are a precise QA assistant.\n"
            f"<|user|>\nPassages:\n{ev_text}\n\nQuestion: {query}\n"
            f"<|assistant|>\nAnswer:")


@torch.inference_mode()
def generate_answer(query: str, evidence: list[str]) -> str:
    prompt  = build_prompt(query, evidence)
    enc     = llm_tok(prompt, return_tensors="pt",
                      max_length=MAX_CTX_LEN, truncation=True).to(DEVICE)
    gen_ids = llm.generate(
        **enc, max_new_tokens=MAX_GEN_LEN,
        do_sample=False, pad_token_id=llm_tok.eos_token_id)
    return llm_tok.decode(
        gen_ids[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1."""
    p_toks = set(pred.lower().split())
    g_toks = set(gold.lower().split())
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline(name: str, route_fn) -> dict:
    """
    route_fn: (idx, item, q_emb) -> source_idx (int)
    Returns dict with EM, F1, routing accuracy, per-source breakdown.
    """
    print(f"\n--- Baseline: {name} ---")
    ems, f1s, route_correct = [], [], []
    per_src_em = {src: [] for src in SOURCES}

    for idx, (item, q_emb) in enumerate(
            tqdm(zip(test_data, test_embs), total=len(test_data), desc=name)):

        src_idx  = route_fn(idx, item, q_emb)
        src_name = SOURCES[src_idx]

        cids     = retrieve(src_name, q_emb)
        evidence = [get_chunk_text(src_name, c) for c in cids]
        pred     = generate_answer(item["query"], evidence)

        em = exact_match(pred, item["answer"])
        f1 = compute_f1(pred, item["answer"])
        ems.append(em)
        f1s.append(f1)
        route_correct.append(float(src_idx == item["label"]))
        per_src_em[src_name].append(em)

    result = {
        "em":           float(np.mean(ems)),
        "f1":           float(np.mean(f1s)),
        "route_acc":    float(np.mean(route_correct)),
        "per_source_em": {src: float(np.mean(v)) if v else None
                          for src, v in per_src_em.items()},
    }
    print(f"  EM={result['em']:.4f}  F1={result['f1']:.4f}  "
          f"RouteAcc={result['route_acc']:.4f}")
    for src, em in result["per_source_em"].items():
        if em is not None:
            print(f"    {src:10s}: EM={em:.4f}")
    return result


# ---------------------------------------------------------------------------
# B3: Train a standalone MLP classifier (no gradient from answer quality)
# Matches RAGROUTE-style disjoint pipeline.
# ---------------------------------------------------------------------------

class StandaloneClassifier(nn.Module):
    """Same architecture as MLPSourceFormer but trained only with L_route."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 128),       nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, K),
        )
    def forward(self, x): return self.net(x)


def train_standalone_classifier(train_embs, train_labels, val_embs, val_labels,
                                 epochs=30, lr=1e-3, seed=42):
    """Trains the disjoint classifier. Returns trained model."""
    torch.manual_seed(seed)
    clf = StandaloneClassifier().to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    ds  = torch.utils.data.TensorDataset(
        torch.from_numpy(train_embs).float(),
        torch.tensor(train_labels, dtype=torch.long))
    dl  = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    best_acc = 0.0
    best_state = None
    patience, p_ctr = 5, 0

    for ep in range(epochs):
        clf.train()
        for embs_b, labs_b in dl:
            embs_b, labs_b = embs_b.to(DEVICE), labs_b.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(clf(embs_b), labs_b).backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            val_t   = torch.from_numpy(val_embs).float().to(DEVICE)
            val_l   = torch.tensor(val_labels, dtype=torch.long).to(DEVICE)
            val_acc = (clf(val_t).argmax(-1) == val_l).float().mean().item()

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            p_ctr      = 0
        else:
            p_ctr += 1
        if p_ctr >= patience:
            break

    clf.load_state_dict(best_state)
    print(f"  Standalone classifier val_acc={best_acc:.4f}")
    return clf


print("\nTraining B3 standalone classifier (no answer gradient)...")
# Use same train/val split from EMB cache
train_embs_np = np.load(Path("query_emb_cache") / "joint_train_embs.npy")
val_embs_np   = np.load(Path("query_emb_cache") / "joint_val_embs.npy")
# Labels come from test_data split used in joint training (80/20 of dev)
random.seed(42)
all_items = list(zip(test_data, test_embs))
random.shuffle(all_items)
split_n = int(0.8 * len(all_items))
tr_labels = [d["label"] for d, _ in all_items[:split_n]]
vl_labels = [d["label"] for d, _ in all_items[split_n:]]

standalone_clf = train_standalone_classifier(
    train_embs_np, tr_labels, val_embs_np, vl_labels)
standalone_clf.eval()


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------

# B3 routing function using standalone classifier
@torch.inference_mode()
def classifier_route(idx, item, q_emb):
    emb_t = torch.from_numpy(q_emb).float().unsqueeze(0).to(DEVICE)
    return standalone_clf(emb_t).argmax(-1).item()

results = {}

results["B1_random"]     = run_baseline("B1: Random Routing",
    lambda idx, item, q: random.randint(0, K - 1))

results["B2_oracle"]     = run_baseline("B2: Oracle Routing",
    lambda idx, item, q: item["label"])

results["B3_classifier"] = run_baseline("B3: Standalone Classifier (RAGROUTE-style)",
    classifier_route)

results["B4_majority"]   = run_baseline("B4: Majority Routing",
    lambda idx, item, q: majority_src)


# ---------------------------------------------------------------------------
# Load joint model (B5) for comparison
# ---------------------------------------------------------------------------

print("\n--- Loading B5: Joint Model (ours) ---")
best_joint_ckpt = CKPT_DIR / f"joint_seed{BEST_SEED}_best.pt"
joint_sf = build_sourceformer("mlp").to(DEVICE)
if best_joint_ckpt.exists():
    ckpt = torch.load(best_joint_ckpt, map_location=DEVICE)
    joint_sf.load_state_dict(ckpt["sf_state_dict"])
    joint_sf.eval()

    @torch.inference_mode()
    def joint_route(idx, item, q_emb):
        emb_t = torch.from_numpy(q_emb).float().unsqueeze(0).to(DEVICE)
        return joint_sf(emb_t).argmax(-1).item()

    results["B5_joint"] = run_baseline("B5: Joint (ours)", joint_route)
else:
    print(f"  Joint checkpoint not found at {best_joint_ckpt}. Skipping B5.")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("BASELINE COMPARISON")
print(f"{'='*70}")
print(f"{'Baseline':<35} {'EM':>6} {'F1':>6} {'RouteAcc':>9}")
print(f"{'-'*70}")
for name, res in results.items():
    print(f"  {name:<33} {res['em']:>6.4f} {res['f1']:>6.4f} {res['route_acc']:>9.4f}")
print(f"{'='*70}")

# Save
with open("baselines_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved baselines_results.json")

# Gap analysis
if "B2_oracle" in results and "B5_joint" in results:
    oracle_em = results["B2_oracle"]["em"]
    joint_em  = results["B5_joint"]["em"]
    clf_em    = results["B3_classifier"]["em"]
    gap_to_oracle = oracle_em - joint_em
    gain_over_clf = joint_em - clf_em
    print(f"\nKey numbers for paper:")
    print(f"  Joint vs Classifier gain: +{gain_over_clf:.4f} EM "
          f"({gain_over_clf/clf_em*100:.1f}% relative)")
    print(f"  Gap to oracle:            -{gap_to_oracle:.4f} EM")
