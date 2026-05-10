import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "../models/phi3-medical"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading base model")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Loading LoRA adapter")

model = PeftModel.from_pretrained(model, adapter_path)

prompt = """
### Question:
What are the symptoms of hypertension?

### Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))