"""
bertscore_evaluation.py  v2
"""

import gc
import json
import torch
import numpy as np
import faiss
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

BASE_MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH      = "../models/phi3-medical"
EVAL_SAMPLES      = 100
MAX_NEW_TOKENS    = 150
TOP_K             = 3
MAX_CONTEXT_CHARS = 3200
MAX_PROMPT_CHARS  = 2800

print("Loading dataset...")
with open("../data/processed/test.json") as f:
    dataset = json.load(f)
dataset = dataset[:EVAL_SAMPLES]
print(f"Dataset loaded: {len(dataset)} samples")

print("Loading embedding model and FAISS index...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index       = faiss.read_index("../models/medical_vector.index")
with open("../models/medical_texts.json") as f:
    texts = json.load(f)
print(f"FAISS index: {index.ntotal} vectors, {index.d}-dim")

print("Loading LLM (fine-tuned)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ft_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)
model.eval()
print("Model loaded.")


def generate(prompt):
    try:
        inputs    = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
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
        print(f"\n  [SKIP] {str(e)[:80]}")
        return "generation_error"


def retrieve_context(question):
    q_vec = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k=TOP_K)
    passages = [texts[i] for i in I[0] if i < len(texts)]
    context, char_budget = "", MAX_CONTEXT_CHARS
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
    return f"<|user|>\n{question}<|end|>\n<|assistant|>\n"


def make_rag_prompt(question, context):
    prompt = (
        f"<|user|>\n"
        f"You are a medical AI assistant. Use the context below to answer accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )
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


pred_ft   = []
pred_rag  = []
refs      = []
questions = []

for item in tqdm(dataset, desc="Inference"):
    text = item["text"]
    try:
        question = text.split("### Question:")[1].split("### Answer:")[0].strip()
        answer   = text.split("### Answer:")[1].strip()
    except IndexError:
        continue
    refs.append(answer)
    questions.append(question)
    context = retrieve_context(question)
    pred_ft.append( generate(make_ft_prompt(question)))
    pred_rag.append(generate(make_rag_prompt(question, context)))

print(f"\nInference complete: {len(refs)} samples")
print(f"FT errors:  {pred_ft.count('generation_error') + pred_ft.count('context_too_long')}")
print(f"RAG errors: {pred_rag.count('generation_error') + pred_rag.count('context_too_long')}")

ERROR_TAGS = {"generation_error", "context_too_long"}

def clean(preds, refs):
    pairs = [(p, r) for p, r in zip(preds, refs) if p not in ERROR_TAGS]
    if not pairs:
        return [], []
    p, r = zip(*pairs)
    return list(p), list(r)

print("\nCalculating BERTScore...")
cp_ft,  cr_ft  = clean(pred_ft,  refs)
cp_rag, cr_rag = clean(pred_rag, refs)

P_ft,  R_ft,  F1_ft  = bert_score(cp_ft,  cr_ft,  lang="en", verbose=False)
P_rag, R_rag, F1_rag = bert_score(cp_rag, cr_rag, lang="en", verbose=False)

print("\nCalculating ROUGE + METEOR...")
rouge_ft  = load("rouge").compute(predictions=cp_ft,  references=cr_ft)
rouge_rag = load("rouge").compute(predictions=cp_rag, references=cr_rag)

meteor_ft  = sum(nltk_meteor([word_tokenize(r)], word_tokenize(p))
                 for p, r in zip(cp_ft,  cr_ft))  / max(len(cp_ft),  1)
meteor_rag = sum(nltk_meteor([word_tokenize(r)], word_tokenize(p))
                 for p, r in zip(cp_rag, cr_rag)) / max(len(cp_rag), 1)

print("\n============================================================")
print(f"BERTScore + ROUGE + METEOR  (n={len(refs)})")
print("============================================================")
print(f"{'':25} {'FT Only':>12} {'FT + RAG':>12}")
print("-" * 52)
print(f"{'BERTScore P':<25} {P_ft.mean().item():>12.4f} {P_rag.mean().item():>12.4f}")
print(f"{'BERTScore R':<25} {R_ft.mean().item():>12.4f} {R_rag.mean().item():>12.4f}")
print(f"{'BERTScore F1':<25} {F1_ft.mean().item():>12.4f} {F1_rag.mean().item():>12.4f}")
print(f"{'ROUGE-1':<25} {rouge_ft['rouge1']:>12.4f} {rouge_rag['rouge1']:>12.4f}")
print(f"{'ROUGE-L':<25} {rouge_ft['rougeL']:>12.4f} {rouge_rag['rougeL']:>12.4f}")
print(f"{'METEOR':<25} {meteor_ft:>12.4f} {meteor_rag:>12.4f}")
print("============================================================")

results = {
    "n_samples": len(refs),
    "fine_tuned": {
        "bertscore_p":  round(P_ft.mean().item(), 4),
        "bertscore_r":  round(R_ft.mean().item(), 4),
        "bertscore_f1": round(F1_ft.mean().item(), 4),
        "rouge1": round(rouge_ft["rouge1"], 4),
        "rougeL": round(rouge_ft["rougeL"], 4),
        "meteor": round(meteor_ft, 4),
    },
    "rag": {
        "bertscore_p":  round(P_rag.mean().item(), 4),
        "bertscore_r":  round(R_rag.mean().item(), 4),
        "bertscore_f1": round(F1_rag.mean().item(), 4),
        "rouge1": round(rouge_rag["rouge1"], 4),
        "rougeL": round(rouge_rag["rougeL"], 4),
        "meteor": round(meteor_rag, 4),
    }
}

with open("bertscore_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("bertscore_predictions.json", "w") as f:
    json.dump([
        {
            "question":  questions[i],
            "reference": refs[i],
            "pred_ft":   pred_ft[i],
            "pred_rag":  pred_rag[i],
        }
        for i in range(min(20, len(refs)))
    ], f, indent=2)

print("\nSaved -> bertscore_results.json")
print("Saved -> bertscore_predictions.json")