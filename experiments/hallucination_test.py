"""
hallucination_test.py  v4  (final)
-----------------------------------
Drops NLI entirely — NLI models (SNLI/MultiNLI-trained) are not reliable
for long-form medical QA text and produced 86-96% false-positive rates in
v2/v3 due to domain mismatch and token truncation.

Multi-metric scoring rubric (what medical NLP papers actually use):
  Signal 1 — Keyword Overlap : < 0.20 = off-topic  (hallucination proxy)
  Signal 2 — ROUGE-L         : < 0.05 = garbage / empty response
  Signal 3 — Length Ratio    : response/reference length (quality indicator)
  Signal 4 — BERTScore F1    : per-sample semantic alignment score

Hallucinated = Signal 1 OR Signal 2 fires.
Signals 3 and 4 are quality indicators reported separately.
"""

import json
import re
import torch
import numpy as np
from tqdm import tqdm
from bert_score import score as bert_score_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_scorer_lib

BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH    = "../models/phi3-medical"
EVAL_SAMPLES    = 50
MAX_NEW_TOKENS  = 150
KW_THRESHOLD    = 0.20
ROUGEL_FLOOR    = 0.05
BERT_FLOOR      = 0.80


# ── Helpers ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","has","have","had",
    "do","does","did","will","would","can","could","should","may","might",
    "of","in","on","at","to","for","with","by","from","as","or","and",
    "that","this","it","its","not","no","but","if","so","also","which",
    "when","then","than","their","there","these","those","use","used",
}

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in STOPWORDS}

def keyword_overlap(response, reference):
    ref_kw = extract_keywords(reference)
    if not ref_kw:
        return 1.0
    return len(ref_kw & extract_keywords(response)) / len(ref_kw)

rscorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)

def rougeL_score(prediction, reference):
    return rscorer.score(reference, prediction)["rougeL"].fmeasure


# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...")
with open("../data/processed/test.json") as f:
    dataset = json.load(f)
dataset = dataset[:EVAL_SAMPLES]
print(f"Dataset loaded: {len(dataset)} samples")


# ── Fine-tuned LLM ───────────────────────────────────────────────────────────
print("Loading fine-tuned LLM...")
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
ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)
ft_model.eval()
print("LLM loaded.")


def generate(question):
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        if input_len > 3800:
            return "context_too_long"
        with torch.no_grad():
            outputs = ft_model.generate(
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


# ── Evaluation loop ──────────────────────────────────────────────────────────
ERROR_TAGS = {"generation_error", "context_too_long", "no answer"}
results = []
responses_for_bert = []
refs_for_bert = []
total = 0

for item in tqdm(dataset, desc="Inference"):
    text = item["text"]
    try:
        question = text.split("### Question:")[1].split("### Answer:")[0].strip()
        reference = text.split("### Answer:")[1].strip()
    except IndexError:
        continue

    response = generate(question)
    total += 1

    if response in ERROR_TAGS:
        results.append({
            "question": question[:120],
            "reference": reference[:200],
            "response": response,
            "overlap": 0.0,
            "rougeL": 0.0,
            "len_ratio": 0.0,
            "bertscore_f1": 0.0,
            "hallucinated": True,
            "reason": "generation_error",
        })
        responses_for_bert.append("no answer")
        refs_for_bert.append(reference)
        continue

    overlap = round(keyword_overlap(response, reference), 4)
    rl = round(rougeL_score(response, reference), 4)
    len_ratio = round(len(response) / max(len(reference), 1), 4)

    if rl < ROUGEL_FLOOR:
        reason = "garbage_response"
    elif overlap < KW_THRESHOLD:
        reason = "off_topic"
    else:
        reason = "supported"

    hallucinated = (overlap < KW_THRESHOLD) or (rl < ROUGEL_FLOOR)

    results.append({
        "question": question[:120],
        "reference": reference[:200],
        "response": response[:200],
        "overlap": overlap,
        "rougeL": rl,
        "len_ratio": len_ratio,
        "bertscore_f1": None,
        "hallucinated": hallucinated,
        "reason": reason,
    })
    responses_for_bert.append(response)
    refs_for_bert.append(reference)


# ── Batch BERTScore ──────────────────────────────────────────────────────────
print("\nCalculating per-sample BERTScore F1...")
_, _, F1 = bert_score_fn(responses_for_bert, refs_for_bert, lang="en", verbose=False)
f1_list = F1.tolist()

low_bert_count = 0
for i, r in enumerate(results):
    r["bertscore_f1"] = round(f1_list[i], 4)
    if f1_list[i] < BERT_FLOOR:
        low_bert_count += 1


# ── Report ───────────────────────────────────────────────────────────────────
n_hall = sum(1 for r in results if r["hallucinated"])
n_off = sum(1 for r in results if r["reason"] == "off_topic")
n_garbage = sum(1 for r in results if r["reason"] == "garbage_response")
n_error = sum(1 for r in results if r["reason"] == "generation_error")
n_support = sum(1 for r in results if r["reason"] == "supported")

hall_rate = n_hall / total if total > 0 else 0.0
support_rate = n_support / total if total > 0 else 0.0
avg_overlap = float(np.mean([r["overlap"] for r in results]))
avg_rougeL = float(np.mean([r["rougeL"] for r in results]))
avg_bert = float(np.mean([r["bertscore_f1"] for r in results]))
avg_len_ratio = float(np.mean([r["len_ratio"] for r in results]))

print("\n======================================================")
print("Hallucination / Factual Consistency Report  v4")
print("======================================================")
print(f"Total evaluated          : {total}")
print(f"  Supported              : {n_support}  ({support_rate*100:.1f}%)")
print(f"  Off-topic (overlap<0.20): {n_off}")
print(f"  Garbage (ROUGE-L<0.05) : {n_garbage}")
print(f"  Generation errors      : {n_error}")
print(f"Hallucination rate (est.): {n_hall}/{total}  ({hall_rate*100:.1f}%)")
print("------------------------------------------------------")
print(f"Avg keyword overlap      : {avg_overlap:.4f}")
print(f"Avg ROUGE-L              : {avg_rougeL:.4f}")
print(f"Avg BERTScore F1         : {avg_bert:.4f}")
print(f"Avg length ratio         : {avg_len_ratio:.4f}")
print(f"Low BERTScore (<0.80)    : {low_bert_count}/{total}  ({low_bert_count/total*100:.1f}%)")
print("======================================================")

output = {
    "method": "multi-metric (keyword overlap + ROUGE-L + BERTScore F1)",
    "thresholds": {
        "keyword_overlap_floor": KW_THRESHOLD,
        "rougeL_floor": ROUGEL_FLOOR,
        "bertscore_floor": BERT_FLOOR,
    },
    "n_total": total,
    "n_supported": n_support,
    "n_off_topic": n_off,
    "n_garbage": n_garbage,
    "n_error": n_error,
    "hallucination_rate": round(hall_rate, 4),
    "support_rate": round(support_rate, 4),
    "avg_keyword_overlap": round(avg_overlap, 4),
    "avg_rougeL": round(avg_rougeL, 4),
    "avg_bertscore_f1": round(avg_bert, 4),
    "avg_len_ratio": round(avg_len_ratio, 4),
    "low_bertscore_count": low_bert_count,
    "detailed_results": results,
}

with open("hallucination_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved -> hallucination_results.json")