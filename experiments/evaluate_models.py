"""
evaluate_models.py
Quick qualitative check — prints model answers for 5 samples.
Fixed: response extraction now uses input_len instead of string splitting.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

print("Script started")

# -----------------------
# Load dataset
# -----------------------
with open("../data/processed/test.json") as f:
    dataset = json.load(f)

dataset = dataset[:5]
print(f"Dataset loaded: {len(dataset)} samples")

# -----------------------
# Load model
# -----------------------
base_model_name = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, "../models/phi3-medical")
model.eval()

print("Model loaded.")

# -----------------------
# Inference loop
# -----------------------
for i, item in enumerate(tqdm(dataset)):

    text = item["text"]

    try:
        question  = text.split("### Question:")[1].split("### Answer:")[0].strip()
        reference = text.split("### Answer:")[1].strip()
    except IndexError:
        continue

    prompt = f"### Question:\n{question}\n\n### Answer:\n"

    inputs    = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]  # FIX: track prompt length

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    # FIX: decode only new tokens — not the full prompt
    new_tokens = outputs[0][input_len:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"\n--- Sample {i+1} ---")
    print(f"Question  : {question[:150]}")
    print(f"Reference : {reference[:150]}")
    print(f"Predicted : {response[:150]}")

print("\nEvaluation finished.")