"""
rag_vs_ft_evaluation.py
-----------------------
Ablation study — THREE configurations:
  1. Base Phi-3       — no fine-tuning, no RAG
  2. Fine-tuned Phi-3 — QLoRA only, no RAG
  3. Fine-tuned + RAG — full system

Metrics: BLEU, ROUGE-1, ROUGE-L, METEOR, BERTScore
Sample size: 200

Fixes:
  - Sequential model loading (one model in VRAM at a time)
  - RAG context truncated to MAX_CONTEXT_TOKENS to prevent 4096 overflow
  - try/except around each generate() call — skips instead of crashing
  - Partial results saved after each phase for safety
"""

import gc
import json
import torch
import faiss
import numpy as np
import nltk
from tqdm import tqdm
from evaluate import load
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.tokenize import word_tokenize

nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt_tab", quiet=True)

print("Starting evaluation...")

BASE_MODEL_NAME    = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH       = "../models/phi3-medical"
TEST_FILE          = "../data/processed/test.json"
VECTOR_INDEX       = "../models/medical_vector.index"
VECTOR_TEXTS       = "../models/medical_texts.json"
MAX_SAMPLES        = 200
MAX_NEW_TOKENS     = 150
TOP_K              = 3
MAX_CONTEXT_TOKENS = 800   # hard cap on retrieved context (~chars/4)
MAX_PROMPT_CHARS   = 2800  # safety cap on total prompt length in characters

with open(TEST_FILE) as f:
    dataset = json.load(f)
dataset = dataset[:MAX_SAMPLES]
print(f"Dataset loaded: {len(dataset)} samples")

bleu_metric  = load("bleu")
rouge_metric = load("rouge")

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading vector database...")
index = faiss.read_index(VECTOR_INDEX)
with open(VECTOR_TEXTS) as f:
    texts = json.load(f)
print(f"FAISS index: {index.ntotal} vectors, {index.d}-dim")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  VRAM freed. Usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")


def generate(model, prompt):
    """Generate a response. Returns error tag string on any failure."""
    try:
        inputs    = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # hard guard — skip if prompt alone is too long
        if input_len > 3800:
            return "context_too_long"

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        resp = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return resp if resp else "no answer"

    except RuntimeError as e:
        torch.cuda.empty_cache()
        print(f"\n  [SKIP] RuntimeError: {str(e)[:80]}")
        return "generation_error"


def retrieve_context(question):
    q_vec = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k=TOP_K)
    passages = [texts[i] for i in I[0] if i < len(texts)]

    # truncate — hard cap total context at MAX_CONTEXT_TOKENS * 4 characters
    char_budget = MAX_CONTEXT_TOKENS * 4
    context = ""
    for p in passages:
        if len(context) + len(p) <= char_budget:
            context += p + "\n\n"
        else:
            remaining = char_budget - len(context)
            if remaining > 100:
                context += p[:remaining]
            break
    return context.strip()


def make_ft_prompt(question):
    return (
        f"<|user|>\n"
        f"{question}<|end|>\n"
        f"<|assistant|>\n"
    )


def make_rag_prompt(question, context):
    prompt = (
        f"<|user|>\n"
        f"You are a medical AI assistant. Use the context below to answer accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )
    # secondary safety trim if still over char budget
    if len(prompt) > MAX_PROMPT_CHARS:
        trim    = len(prompt) - MAX_PROMPT_CHARS
        context = context[:-trim] if trim < len(context) else context[:200]
        prompt  = (
            f"<|user|>\n"
            f"You are a medical AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}<|end|>\n"
            f"<|assistant|>\n"
        )
    return prompt


# pre-parse dataset and fetch all contexts (CPU only — no GPU needed)
parsed = []
for item in dataset:
    text = item["text"]
    try:
        question = text.split("### Question:")[1].split("### Answer:")[0].strip()
        answer   = text.split("### Answer:")[1].strip()
        parsed.append((question, answer))
    except IndexError:
        continue

questions = [p[0] for p in parsed]
refs      = [p[1] for p in parsed]
contexts  = [retrieve_context(q) for q in tqdm(questions, desc="Pre-fetching contexts")]

avg_chars = sum(len(c) for c in contexts) / len(contexts)
print(f"Parsed {len(parsed)} valid samples.")
print(f"Average context length: {avg_chars:.0f} chars (~{avg_chars/4:.0f} tokens)")


# ── PHASE 1: Base model inference ────────────────────────────────────────────
print("\n[Phase 1] Loading BASE model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

pred_base = []
for q in tqdm(questions, desc="Base model inference"):
    pred_base.append(generate(model, make_ft_prompt(q)))

errs = pred_base.count("generation_error") + pred_base.count("context_too_long")
print(f"Base inference done. Errors/skips: {errs}/{len(pred_base)}")
free_model(model)

with open("phase1_base_preds.json", "w") as f:
    json.dump(pred_base, f)
print("Phase 1 predictions saved → phase1_base_preds.json")


# ── PHASE 2: Fine-tuned model inference ──────────────────────────────────────
print("\n[Phase 2] Loading FINE-TUNED model...")
ft_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)
ft_model.eval()

pred_ft  = []
pred_rag = []
for q, ctx in tqdm(zip(questions, contexts), total=len(questions), desc="FT + RAG inference"):
    pred_ft.append( generate(ft_model, make_ft_prompt(q)))
    pred_rag.append(generate(ft_model, make_rag_prompt(q, ctx)))

ft_errs  = pred_ft.count("generation_error")
rag_errs = pred_rag.count("generation_error") + pred_rag.count("context_too_long")
print(f"FT errors: {ft_errs} | RAG errors/skips: {rag_errs}")
free_model(ft_model)

with open("phase2_ft_preds.json", "w") as f:
    json.dump({"ft": pred_ft, "rag": pred_rag}, f)
print("Phase 2 predictions saved → phase2_ft_preds.json")


# save sample predictions for manual review
with open("predictions_sample.json", "w") as f:
    json.dump([
        {
            "question":  questions[i],
            "reference": refs[i],
            "base":      pred_base[i],
            "ft":        pred_ft[i],
            "rag":       pred_rag[i],
        }
        for i in range(min(20, len(refs)))
    ], f, indent=2)


# ── Metrics ───────────────────────────────────────────────────────────────────
ERROR_TAGS = {"generation_error", "context_too_long"}

def compute_metrics(preds, refs, label):
    print(f"  Computing: {label}")
    clean  = [(p, r) for p, r in zip(preds, refs) if p not in ERROR_TAGS]
    if not clean:
        return {"label": label, "n_scored": 0,
                "bleu": 0, "rouge1": 0, "rougeL": 0, "meteor": 0, "bertscore": 0}
    cp, cr = zip(*clean)
    cp, cr = list(cp), list(cr)

    b  = bleu_metric.compute(predictions=cp, references=[[r] for r in cr])
    r  = rouge_metric.compute(predictions=cp, references=cr)
    _, _, F1 = bert_score(cp, cr, lang="en", verbose=False)
    meteors = [
        nltk_meteor([word_tokenize(ref)], word_tokenize(p))
        for p, ref in zip(cp, cr)
    ]
    print(f"    Scored {len(cp)}/{len(preds)} ({len(preds)-len(cp)} errors excluded)")
    return {
        "label":     label,
        "n_scored":  len(cp),
        "bleu":      round(b["bleu"], 4),
        "rouge1":    round(r["rouge1"], 4),
        "rougeL":    round(r["rougeL"], 4),
        "meteor":    round(sum(meteors) / len(meteors), 4),
        "bertscore": round(float(F1.mean()), 4),
    }


print("\nCalculating metrics...")
res_base = compute_metrics(pred_base, refs, "Base Phi-3 (no FT, no RAG)")
res_ft   = compute_metrics(pred_ft,   refs, "Fine-tuned Phi-3")
res_rag  = compute_metrics(pred_rag,  refs, "Fine-tuned Phi-3 + RAG")

cols   = ["bleu", "rouge1", "rougeL", "meteor", "bertscore"]
header = f"{'Model':<32}" + "".join(f"{c:>12}" for c in cols)
sep    = "=" * len(header)

print("\n" + sep)
print(f"ABLATION STUDY RESULTS  (n={len(refs)})")
print(sep)
print(header)
print("-" * len(header))
for res in [res_base, res_ft, res_rag]:
    row = f"{res['label']:<32}" + "".join(f"{res[c]:>12.4f}" for c in cols)
    print(row)
print(sep)

print("\nImprovement (Fine-tuned vs Base):")
for c in cols:
    diff  = res_ft[c] - res_base[c]
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
    print(f"  {c:<12}: {diff:+.4f} {arrow}")

print("\nImprovement (FT+RAG vs Fine-tuned):")
for c in cols:
    diff  = res_rag[c] - res_ft[c]
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
    print(f"  {c:<12}: {diff:+.4f} {arrow}")

output = {
    "n_samples": len(refs),
    "results":   [res_base, res_ft, res_rag],
    "improvements": {
        "ft_vs_base": {c: round(res_ft[c]  - res_base[c], 4) for c in cols},
        "rag_vs_ft":  {c: round(res_rag[c] - res_ft[c],   4) for c in cols},
    },
    "sample_predictions": [
        {
            "question":  questions[i],
            "reference": refs[i],
            "base":      pred_base[i],
            "ft":        pred_ft[i],
            "rag":       pred_rag[i],
        }
        for i in range(min(20, len(refs)))
    ]
}

with open("ablation_results.json", "w") as f:
    json.dump(output, f, indent=2)
with open("full_evaluation_results.json", "w") as f:
    json.dump(output["results"], f, indent=2)

print("\nSaved → ablation_results.json")
print("Saved → full_evaluation_results.json")
print("Evaluation complete.")