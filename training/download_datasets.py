from datasets import load_dataset
import json
import os
import requests
import zipfile
import io

os.makedirs("../data/raw", exist_ok=True)

# -------------------------
# PubMedQA
# -------------------------
print("Downloading PubMedQA...")
pubmedqa = load_dataset("pubmed_qa", "pqa_labeled")

with open("../data/raw/pubmedqa.json", "w") as f:
    json.dump(pubmedqa["train"].to_list(), f)

print("PubMedQA saved")

# -------------------------
# MedQA (USMLE)
# -------------------------
print("Downloading MedQA...")
medqa = load_dataset("GBaker/MedQA-USMLE-4-options")

with open("../data/raw/medqa.json", "w") as f:
    json.dump(medqa["train"].to_list(), f)

print("MedQA saved")

# -------------------------
# MedQuAD (GitHub)
# -------------------------
print("Downloading MedQuAD from GitHub...")

url = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
r = requests.get(url)

z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("../data/raw")

print("MedQuAD downloaded and extracted")

print("All datasets downloaded successfully!")