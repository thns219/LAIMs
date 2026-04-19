import os
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ==================================================
# CONFIG
# ==================================================
SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# PATH
# ==================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "diffusion_data")
RESULT_DIR = os.path.join(BASE_DIR, "result")

os.makedirs(RESULT_DIR, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================
image_paths = []
labels = []

label_map = {
    "0_real": 0,
    "1_fake": 1
}

for folder_name, label_id in label_map.items():
    folder = os.path.join(DATA_DIR, folder_name)

    if not os.path.exists(folder):
        raise ValueError(f"Missing folder: {folder}")

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(folder, file))
            labels.append(label_id)

print("=" * 50)
print("TOTAL IMAGES :", len(image_paths))


# ==================================================
# SPLIT DATA
# ==================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("TRAIN :", len(X_train))
print("VAL   :", len(X_val))
print("TEST  :", len(X_test))
print("=" * 50)


# ==================================================
# DATASET
# ==================================================
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


# ==================================================
# TRANSFORM
# ==================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==================================================
# DATALOADER
# ==================================================
train_dataset = ImageDataset(X_train, y_train, train_transform)
val_dataset = ImageDataset(X_val, y_val, test_transform)
test_dataset = ImageDataset(X_test, y_test, test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==================================================
# MODEL
# ==================================================
model = models.resnet50(weights=None)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

model = model.to(DEVICE)


# ==================================================
# LOSS / OPTIMIZER
# ==================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)


# ==================================================
# TRAIN
# ==================================================
best_acc = 0
best_model = copy.deepcopy(model.state_dict())

print("\nSTART TRAINING...\n")

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels_batch in train_loader:

        images = images.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # VALIDATION
    model.eval()

    preds = []
    truths = []

    with torch.no_grad():
        for images, labels_batch in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(pred)
            truths.extend(labels_batch.numpy())

    val_acc = accuracy_score(truths, preds)

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_model = copy.deepcopy(model.state_dict())


# ==================================================
# SAVE MODEL
# ==================================================
torch.save(
    best_model,
    os.path.join(BASE_DIR, "resnet_best.pth")
)


# ==================================================
# TEST
# ==================================================
model.load_state_dict(best_model)
model.eval()

preds = []
truths = []

with torch.no_grad():
    for images, labels_batch in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        pred = torch.argmax(outputs, dim=1).cpu().numpy()

        preds.extend(pred)
        truths.extend(labels_batch.numpy())


# ==================================================
# METRICS
# ==================================================
acc = accuracy_score(truths, preds)
precision = precision_score(truths, preds)
recall = recall_score(truths, preds)
f1 = f1_score(truths, preds)

print("\n" + "=" * 50)
print("FINAL TEST RESULT")
print("=" * 50)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("=" * 50)

report = classification_report(
    truths,
    preds,
    target_names=["Real", "Fake"]
)

print(report)


# ==================================================
# SAVE TXT
# ==================================================
with open(
    os.path.join(RESULT_DIR, "resnet_report.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)
    f.write(f"\nAccuracy : {acc:.4f}")
    f.write(f"\nPrecision: {precision:.4f}")
    f.write(f"\nRecall   : {recall:.4f}")
    f.write(f"\nF1-score : {f1:.4f}")


# ==================================================
# SAVE CSV
# ==================================================
df = pd.DataFrame(
    classification_report(
        truths,
        preds,
        target_names=["Real", "Fake"],
        output_dict=True
    )
).transpose()

df.to_csv(
    os.path.join(RESULT_DIR, "resnet_report.csv")
)


# ==================================================
# PROFESSIONAL CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(truths, preds)

labels_name = ["Real", "Fake"]

cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

annot = np.empty_like(cm).astype(str)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i,j]}\n({cm_percent[i,j]*100:.1f}%)"

sns.set_theme(style="white")

plt.figure(figsize=(8, 6), facecolor="white")

ax = sns.heatmap(
    cm,
    annot=annot,
    fmt="",
    cmap="Blues",
    linewidths=1.2,
    linecolor="#EAEAEA",
    square=True,
    cbar=True,
    xticklabels=labels_name,
    yticklabels=labels_name,
    annot_kws={
        "fontsize": 13,
        "fontweight": "normal",
        "family": "Arial"
    }
)

plt.title(
    "Confusion Matrix - ResNet50",
    fontsize=18,
    fontweight="bold",
    pad=18
)

plt.xlabel(
    "Predicted Label",
    fontsize=13,
    fontweight="semibold"
)

plt.ylabel(
    "True Label",
    fontsize=13,
    fontweight="semibold"
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_edgecolor("#DDDDDD")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "resnet_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==================================================
# SUMMARY CSV
# ==================================================
summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [acc, precision, recall, f1]
})

summary.to_csv(
    os.path.join(RESULT_DIR, "resnet_summary.csv"),
    index=False
)

print("\nALL RESULTS SAVED TO /result/")