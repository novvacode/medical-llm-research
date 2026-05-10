"""
evaluate_model.py
-----------------
MCQ accuracy evaluation — v4.

Fixes:
  1. Phi-3 native chat template
  2. Answer priming ("The correct answer is ")
  3. Multi-strategy letter extractor
  4. Text-to-option fallback for unanswered cases
  5. Option shuffling at inference to break position/A bias
  6. nf4 + double quantisation
  7. Saves error cases + prediction distribution for paper
"""

import json
import torch
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# reproducibility
random.seed(42)

# config
BASE_MODEL   = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "../models/phi3-medical"
TEST_FILE    = "../data/processed/test.json"
EVAL_SAMPLES = 100

# load model
print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("Model loaded.")

# load test data
with open(TEST_FILE) as f:
    dataset = json.load(f)

mcq_items = [d for d in dataset if "Options:" in d["text"]]
mcq_items = mcq_items[:EVAL_SAMPLES]
print(f"MCQ items found: {len(mcq_items)}")

if len(mcq_items) == 0:
    print("No MCQ items found. Check test.json format.")
    exit()


def shuffle_options(question_block, ref_letter):
    """
    Randomly reorder A/B/C/D options and remap the reference letter.
    Neutralises position/A bias at inference time.
    """
    if "Options:" not in question_block:
        return question_block, ref_letter

    stem, opts_raw = question_block.split("Options:", 1)
    options = re.findall(r"([A-D]):\s*(.+?)(?=\n[A-D]:|$)", opts_raw, re.DOTALL)
    if len(options) < 2:
        return question_block, ref_letter

    opt_dict = {letter: text.strip() for letter, text in options}
    correct_text = opt_dict.get(ref_letter, "")

    opt_texts = [text.strip() for _, text in options]
    random.shuffle(opt_texts)

    letters = ["A", "B", "C", "D"]
    new_opts = ""
    new_ref = ref_letter

    for i, text in enumerate(opt_texts):
        new_opts += f"{letters[i]}: {text}\n"
        if text == correct_text:
            new_ref = letters[i]

    new_block = f"{stem}Options:\n{new_opts.strip()}"
    return new_block, new_ref


def extract_letter(text):
    t = text.strip()

    m = re.match(r"^([A-D])[.\s\):]", t)
    if m:
        return m.group(1)

    m = re.search(r"answer[:\s]+(?:is[:\s]*)?([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"(?:option|choice)[:\s]+([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"\b([A-D])\b", t[:80])
    if m:
        return m.group(1)

    return "?"


def match_text_to_option(response, question_block):
    """
    Fallback: if model outputs option text instead of a letter,
    match it against the option texts.
    """
    options = re.findall(r"([A-D]):\s*(.+?)(?=\n[A-D]:|$)", question_block, re.DOTALL)
    resp_lower = response.lower().strip()

    best_letter = "?"
    best_score = 0

    for letter, option_text in options:
        words = [w for w in option_text.lower().split() if len(w) > 3]
        score = sum(1 for w in words if w in resp_lower)

        if score > best_score:
            best_score = score
            best_letter = letter

    return best_letter if best_score > 0 else "?"


# evaluation loop
correct = 0
total = 0
results = []
errors = []
pred_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "?": 0}

for item in tqdm(mcq_items, desc="Evaluating MCQ"):
    text = item["text"]

    try:
        question_block = text.split("### Question:")[1].split("### Answer:")[0].strip()
        reference = text.split("### Answer:")[1].strip()
    except IndexError:
        continue

    ref_match = re.match(r"^([A-D])", reference.strip())
    if not ref_match:
        continue
    ref_letter = ref_match.group(1)

    # shuffle options to neutralise A-bias
    question_block, ref_letter = shuffle_options(question_block, ref_letter)

    prompt = (
        f"<|user|>\n"
        f"You are a medical expert taking a multiple choice exam.\n"
        f"Read the question and ALL options carefully before choosing.\n"
        f"Reply with ONLY a single letter: A, B, C, or D. No explanation.\n\n"
        f"{question_block}<|end|>\n"
        f"<|assistant|>\n"
        f"The correct answer is "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pred_letter = extract_letter(response)

    if pred_letter == "?":
        pred_letter = match_text_to_option(response, question_block)

    pred_dist[pred_letter] = pred_dist.get(pred_letter, 0) + 1

    is_correct = (pred_letter == ref_letter)
    if is_correct:
        correct += 1
    total += 1

    record = {
        "question": question_block[:150],
        "reference": ref_letter,
        "predicted": pred_letter,
        "raw_output": response[:100],
        "correct": is_correct,
    }
    results.append(record)

    if not is_correct:
        errors.append(record)

# report
accuracy = correct / total if total > 0 else 0.0
unanswered = pred_dist.get("?", 0)

print("\n============================")
print("MCQ Accuracy Results")
print("============================")
print(f"Total MCQ items evaluated : {total}")
print(f"Correct predictions       : {correct}")
print(f"Accuracy                  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Unanswered (?)            : {unanswered}")
print(f"\nPrediction distribution   : {pred_dist}")

with open("mcq_accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("mcq_errors.json", "w") as f:
    json.dump(errors, f, indent=2)

print("\nResults → mcq_accuracy_results.json")
print(f"Errors  → mcq_errors.json ({len(errors)} wrong)")