import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# PATH
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
RESULT_DIR = os.path.join(ROOT_DIR, "result")

# ==================================================
# LOAD METRICS
# ==================================================
def load_metrics(path):
    df = pd.read_csv(path, index_col=0)

    accuracy = float(df.loc["accuracy", "precision"])
    precision = float(df.loc["macro avg", "precision"])
    recall = float(df.loc["macro avg", "recall"])
    f1 = float(df.loc["macro avg", "f1-score"])

    return [accuracy, precision, recall, f1]


bert = load_metrics(os.path.join(RESULT_DIR, "bert_report.csv"))
roberta = load_metrics(os.path.join(RESULT_DIR, "roberta_report.csv"))
vit = load_metrics(os.path.join(RESULT_DIR, "vit_report.csv"))
resnet = load_metrics(os.path.join(RESULT_DIR, "resnet_report.csv"))

# ==================================================
# DATAFRAME
# ==================================================
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

df_compare = pd.DataFrame({
    "BERT": bert,
    "RoBERTa": roberta,
    "ViT": vit,
    "ResNet50": resnet
}, index=metrics)

# ==================================================
# STYLE
# ==================================================
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

# ==================================================
# COLORS
# ==================================================
colors = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#9C7AC7"
]

# ==================================================
# BAR CHART
# ==================================================
x = np.arange(len(metrics))
width = 0.18

ax.bar(x - 1.5*width, df_compare["BERT"], width, label="BERT", color=colors[0])
ax.bar(x - 0.5*width, df_compare["RoBERTa"], width, label="RoBERTa", color=colors[1])
ax.bar(x + 0.5*width, df_compare["ViT"], width, label="ViT", color=colors[2])
ax.bar(x + 1.5*width, df_compare["ResNet50"], width, label="ResNet50", color=colors[3])

# ==================================================
# TITLE
# ==================================================
ax.set_title(
    "Performance Comparison of Multimodal AI Detection Models",
    fontsize=22,
    fontweight="bold",
    pad=45
)

# ==================================================
# LEGEND (DƯỚI TITLE - KHÔNG ĐÈ)
# ==================================================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    ncol=4,
    frameon=False,
    fontsize=13,
    columnspacing=2.0,
    handletextpad=0.6
)

# ==================================================
# AXIS
# ==================================================
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=15, fontweight="semibold")

ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.01, 0.1))

ax.set_ylabel("Score", fontsize=16, fontweight="bold")
ax.set_xlabel("Evaluation Metrics", fontsize=16, fontweight="bold")

# ==================================================
# GRID
# ==================================================
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.grid(axis="x", visible=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ==================================================
# LAYOUT
# ==================================================
plt.subplots_adjust(top=0.82, bottom=0.12)

# ==================================================
# SAVE
# ==================================================
plt.savefig(
    os.path.join(RESULT_DIR, "compare.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("DONE - Legend placed below title professionally.")