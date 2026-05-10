# Medical LLM Research - Phi-3 QLoRA + RAG

Domain-adapted Medical Question Answering using QLoRA fine-tuning and Retrieval-Augmented Generation on a single consumer GPU (NVIDIA RTX 4050, 6 GB VRAM).

## Results

| Metric       | Base Phi-3 | Fine-tuned | FT + RAG |
|--------------|:----------:|:----------:|:--------:|
| BLEU         | 0.038      | 0.134      | 0.151    |
| ROUGE-1      | 0.200      | 0.269      | 0.284    |
| ROUGE-L      | 0.133      | 0.211      | 0.227    |
| METEOR       | 0.173      | 0.236      | 0.242    |
| BERTScore F1 | 0.833      | 0.856      | 0.859    |

QLoRA fine-tuning improves BLEU by +251% over zero-shot baseline. RAG adds a further +14.2% METEOR on top of fine-tuning.

## Setup

Requirements: Python 3.10+, NVIDIA GPU 6GB+ VRAM, CUDA 12.1

Install:
    git clone https://github.com/novvacode/medical-llm-research.git
    cd medical-llm-research
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Usage

1. Download datasets:   python training/download_datasets.py
2. Preprocess:          python training/process_datasets.py
3. Split:               python training/split_dataset.py
4. Fine-tune:           python training/train_qlora.py
5. Build RAG index:     python rag/build_vector_db.py
6. Ask a question:      python rag/ask_medical_question.py
7. Evaluate:            python experiments/bertscore_evaluation.py
8. Ablation study:      python experiments/rag_vs_ft_evaluation.py
9. Hallucination test:  python experiments/hallucination_test.py

## Model Details

Base Model:        microsoft/Phi-3-mini-4k-instruct
Parameters:        3.82 billion
Quantisation:      4-bit NF4 + double quantisation
LoRA Rank:         16
LoRA Alpha:        32
Trainable Params:  20.5M (0.54% of total)
Training Epochs:   5
Learning Rate:     2e-4 with cosine schedule

## Project Structure

    medical-llm-research/
    training/       QLoRA fine-tuning scripts
    rag/            FAISS vector DB and inference
    experiments/    Ablation, BERTScore, hallucination tests
    evaluation/     Model evaluation scripts
    data/processed/ Train, val, test splits
    notebooks/      Environment and quick tests
    app/            Gradio demo app

## Hardware

GPU:  NVIDIA GeForce RTX 4050 Laptop 6GB GDDR6
CPU:  Intel Core i7-13700HX 16 cores
RAM:  16 GB DDR5
OS:   Windows 11

## Datasets

PubMedQA  - 211k samples - Long-form biomedical QA
MedQA     - 12.7k samples - USMLE multiple choice QA
MedQuAD   - 47.4k samples - Consumer medical QA

## Acknowledgements

Hugging Face Transformers, PEFT, BitsAndBytes, FAISS, SentenceTransformers, TRL
