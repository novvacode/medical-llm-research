import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Data (tumhare results)
# -------------------------

models = ["Fine-tuned", "Fine-tuned + RAG"]

bleu_scores = [0.0, 0.0]
rouge_scores = [0.005, 0.018]
bert_scores = [0.763, 0.783]

sns.set(style="whitegrid")

# -------------------------
# BLEU Graph
# -------------------------

plt.figure()

plt.bar(models, bleu_scores)

plt.title("BLEU Score Comparison")
plt.ylabel("BLEU Score")

plt.savefig("../results_bleu.png")

plt.close()

# -------------------------
# ROUGE Graph
# -------------------------

plt.figure()

plt.bar(models, rouge_scores)

plt.title("ROUGE-L Score Comparison")
plt.ylabel("ROUGE-L")

plt.savefig("../results_rouge.png")

plt.close()

# -------------------------
# BERTScore Graph
# -------------------------

plt.figure()

plt.bar(models, bert_scores)

plt.title("BERTScore Comparison")
plt.ylabel("BERTScore")

plt.savefig("../results_bertscore.png")

plt.close()

print("Graphs generated successfully!")