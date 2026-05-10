import json
import random
import os

input_path = "../data/processed/medical_qa_dataset.json"
output_path = "../data/processed"

os.makedirs(output_path, exist_ok=True)

print("Loading dataset...")

with open(input_path) as f:
    data = json.load(f)

print("Total samples:", len(data))

# Shuffle dataset
random.shuffle(data)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1

train_size = int(len(data) * train_ratio)
val_size = int(len(data) * val_ratio)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

# Save splits
with open(f"{output_path}/train.json", "w") as f:
    json.dump(train_data, f)

with open(f"{output_path}/validation.json", "w") as f:
    json.dump(val_data, f)

with open(f"{output_path}/test.json", "w") as f:
    json.dump(test_data, f)

print("Dataset split completed!")