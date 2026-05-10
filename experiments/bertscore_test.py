import json
from bert_score import score
from tqdm import tqdm

with open("../data/processed/test.json") as f:
    dataset = json.load(f)

dataset = dataset[:100]

references = []
predictions = []

for item in dataset:

    text = item["text"]

    question = text.split("### Question:")[1].split("### Answer:")[0]
    answer = text.split("### Answer:")[1]

    # placeholder prediction
    prediction = answer  

    predictions.append(prediction)
    references.append(answer)

P, R, F1 = score(predictions, references, lang="en")

print("BERTScore F1:", F1.mean().item())