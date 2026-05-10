import json
import os
import xml.etree.ElementTree as ET

# paths
raw_path = "../data/raw"
processed_path = "../data/processed"

os.makedirs(processed_path, exist_ok=True)

final_dataset = []

# -------------------------
# PubMedQA
# -------------------------
print("Processing PubMedQA...")

with open(f"{raw_path}/pubmedqa.json") as f:
    pubmed = json.load(f)

for item in pubmed:

    question = item.get("question")
    answer = item.get("long_answer")

    if question and answer:

        text = f"""### Question:
{question}

### Answer:
{answer}"""

        final_dataset.append({"text": text})

print("PubMedQA processed:", len(pubmed))


# -------------------------
# MedQA
# -------------------------
print("Processing MedQA...")

with open(f"{raw_path}/medqa.json") as f:
    medqa = json.load(f)

for item in medqa:

    question = item.get("question")
    options = item.get("options")
    answer = item.get("answer")

    if question and answer:

        if options:
            opt_text = " ".join([f"{k}: {v}" for k, v in options.items()])
        else:
            opt_text = ""

        text = f"""### Question:
{question}

Options:
{opt_text}

### Answer:
{answer}"""

        final_dataset.append({"text": text})

print("MedQA processed:", len(medqa))


# -------------------------
# MedQuAD
# -------------------------
print("Processing MedQuAD XML...")

medquad_path = f"{raw_path}/MedQuAD-master"

count = 0

for root, dirs, files in os.walk(medquad_path):

    for file in files:

        if file.endswith(".xml"):

            file_path = os.path.join(root, file)

            try:

                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for qa in root_xml.iter():

                    if qa.tag.lower() == "qapair":

                        question = qa.findtext("Question")
                        answer = qa.findtext("Answer")

                        if question and answer:

                            text = f"""### Question:
{question}

### Answer:
{answer}"""

                            final_dataset.append({"text": text})
                            count += 1

            except:
                continue

print("MedQuAD processed:", count)


# -------------------------
# Save dataset
# -------------------------
print("Total samples:", len(final_dataset))

with open(f"{processed_path}/medical_qa_dataset.json", "w") as f:

    json.dump(final_dataset, f, indent=2)

print("Dataset saved successfully!")