import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Loading embedding model")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading dataset")

with open("../data/processed/train.json") as f:
    data = json.load(f)

texts = [x["text"] for x in data[:10000]]

print("Creating embeddings")

embeddings = embed_model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "../models/medical_vector.index")

with open("../models/medical_texts.json","w") as f:
    json.dump(texts,f)

print("Vector DB created!")