import matplotlib.pyplot as plt

# Box positions
boxes = {
    "User Question": (0.5, 0.9),
    "Retriever\n(FAISS Vector DB)": (0.5, 0.7),
    "Relevant Medical\nDocuments": (0.5, 0.5),
    "Fine-Tuned Phi-3\n(QLoRA)": (0.5, 0.3),
    "Final Medical Answer": (0.5, 0.1)
}

plt.figure(figsize=(6,8))

for text, (x,y) in boxes.items():
    plt.text(
        x, y, text,
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.5")
    )

# Draw arrows
plt.arrow(0.5,0.85,0,-0.1,head_width=0.02,length_includes_head=True)
plt.arrow(0.5,0.65,0,-0.1,head_width=0.02,length_includes_head=True)
plt.arrow(0.5,0.45,0,-0.1,head_width=0.02,length_includes_head=True)
plt.arrow(0.5,0.25,0,-0.1,head_width=0.02,length_includes_head=True)

plt.axis('off')

plt.title("Architecture of the Proposed Medical QA System")

plt.savefig("../architecture_diagram.png")

print("Architecture diagram generated!")