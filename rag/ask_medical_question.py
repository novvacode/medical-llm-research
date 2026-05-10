import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Load embedding model ──────────────────────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Load FAISS index ──────────────────────────────────────────────────────────
print("Loading vector DB...")
index = faiss.read_index("../models/medical_vector.index")

with open("../models/medical_texts.json") as f:
    texts = json.load(f)

# ── Load LLM ──────────────────────────────────────────────────────────────────
print("Loading medical LLM...")
base_model = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # FIX: add nf4 quantisation type
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,          # FIX: enable double quantisation
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,                  # FIX: required for Phi-3
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,                  # FIX: required for Phi-3
)

if tokenizer.pad_token is None:              # FIX: prevents padding warning
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, "../models/phi3-medical")
model.eval()                                 # FIX: set eval mode, disables dropout

# ── Get correct stop token IDs for Phi-3 ─────────────────────────────────────
end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

# ── Ask questions loop ────────────────────────────────────────────────────────
while True:
    question = input("\nAsk medical question: ").strip()

    if question.lower() == "exit":
        break

    if not question:                         # FIX: skip empty input
        continue

    # Embed query
    query_vec = embed_model.encode(
        [question],
        normalize_embeddings=True,           # FIX: normalise for cosine similarity
    )

    # Search vector DB
    D, I = index.search(np.array(query_vec, dtype=np.float32), k=3)  # FIX: k=3
    context = "\n\n".join([texts[i] for i in I[0] if i < len(texts)])

    # FIX: use Phi-3 native chat template instead of raw prompt
    prompt = (
        f"<|user|>\n"
        f"You are a helpful medical assistant. "
        f"Use the context below to answer the question accurately.\n\n"
        f"Context:\n{context[:2500]}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )

    inputs    = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]  # FIX: record prompt length

    with torch.no_grad():                     # FIX: no_grad saves VRAM
        outputs = model.generate(
            **inputs,
            max_new_tokens     = 300,
            do_sample          = False,
            repetition_penalty = 1.1,
            eos_token_id       = [eos_token_id, end_token_id],  # FIX: both stop tokens
            pad_token_id       = tokenizer.pad_token_id,
        )

    # FIX: decode ONLY new tokens, not the full prompt
    new_tokens = outputs[0][input_len:]
    answer     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\nAnswer:")                        # FIX: print inside the loop
    print(answer if answer else "No answer generated.")