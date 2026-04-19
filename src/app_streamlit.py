import os
import base64
import torch
import streamlit as st
import numpy as np
from PIL import Image

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

from torchvision import models, transforms

# ===============================
# FIX CUDA MEMORY
# ===============================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Multimodal Detection", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_BERT = os.path.join(ROOT_DIR, "models", "bert_detector")
MODEL_ROBERTA = os.path.join(ROOT_DIR, "models", "roberta_detector")
MODEL_RESNET = os.path.join(ROOT_DIR, "resnet_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# HEADER (GIỮ NGUYÊN)
# ===============================
logo_path = os.path.join(BASE_DIR, "logo_qnu.jpg")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:15px; margin-bottom:20px;">
        <img src="data:image/png;base64,{logo_base64}" width="70">
        <div>
            <div style="font-size:18px; font-weight:bold;">
                Quy Nhon University
            </div>
            <div style="font-size:14px;">
                Information Technology
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# LOAD MODELS (FIX OOM)
# ===============================
@st.cache_resource
def load_models():

    # ===== TEXT MODELS (CPU ONLY) =====
    bert_tok = DistilBertTokenizer.from_pretrained(MODEL_BERT)
    bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_BERT)
    bert_model.eval()

    rob_tok = RobertaTokenizer.from_pretrained(MODEL_ROBERTA)
    rob_model = RobertaForSequenceClassification.from_pretrained(MODEL_ROBERTA)
    rob_model.eval()

    # ===== RESNET =====
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)

    state = torch.load(MODEL_RESNET, map_location="cpu")
    resnet.load_state_dict(state)
    resnet.eval()

    if torch.cuda.is_available():
        resnet = resnet.to("cuda")

    return bert_tok, bert_model, rob_tok, rob_model, resnet

bert_tok, bert_model, rob_tok, rob_model, resnet = load_models()

# ===============================
# LABELS (CHUẨN HÓA)
# ===============================
label_text = {0: "Human", 1: "AI"}
label_img = {0: "REAL", 1: "FAKE"}

# ===============================
# IMAGE TRANSFORM (RESNET)
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===============================
# PREDICT TEXT
# ===============================
def predict_text(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs))

    return pred, probs

# ===============================
# PREDICT IMAGE (RESNET)
# ===============================
def predict_image(image):
    img = transform(image).unsqueeze(0)

    if torch.cuda.is_available():
        img = img.to("cuda")

    with torch.no_grad():
        logits = resnet(img)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    return pred, probs

# ===============================
# UI TABLE
# ===============================
def render_table(results):
    st.markdown("### Results")

    for name, label, conf in results:
        col1, col2, col3 = st.columns([2, 2, 4])

        with col1:
            st.write(name)

        with col2:
            st.write(label)

        with col3:
            st.progress(float(conf) / 100)

        st.caption(f"{conf:.2f}%")

# ===============================
# WEIGHTED FUSION (CHUẨN LUẬN VĂN)
# ===============================
def final_decision(results):

    weights = {
        "BERT": 0.3,
        "RoBERTa": 0.3,
        "ResNet": 0.4
    }

    score_ai = 0
    score_human = 0

    for name, label, conf in results:
        w = weights.get(name, 0.3)
        score = conf * w

        if label in ["AI", "FAKE"]:
            score_ai += score
        else:
            score_human += score

    return "AI GENERATED" if score_ai > score_human else "HUMAN"

# ===============================
# MAIN UI (GIỮ NGUYÊN)
# ===============================
st.title("Multimodal AI Detection System")

mode = st.radio("Mode", ["Text", "Image", "Fusion"])

# ===============================
# TEXT
# ===============================
if mode == "Text":
    text = st.text_area("Input text")

    if st.button("Run"):
        if text.strip():

            p1, prob1 = predict_text(bert_model, bert_tok, text)
            p2, prob2 = predict_text(rob_model, rob_tok, text)

            results = [
                ("BERT", label_text[p1], prob1[p1] * 100),
                ("RoBERTa", label_text[p2], prob2[p2] * 100)
            ]

            render_table(results)
            st.markdown(f"**Final:** {final_decision(results)}")

        else:
            st.warning("Please enter text")

# ===============================
# IMAGE (RESNET)
# ===============================
elif mode == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Run"):

            p, prob = predict_image(img)

            results = [
                ("ResNet", label_img[p], prob[p] * 100)
            ]

            render_table(results)

# ===============================
# FUSION (BERT + ROBERTA + RESNET)
# ===============================
elif mode == "Fusion":
    text = st.text_area("Text")
    file = st.file_uploader("Image", type=["jpg", "png", "jpeg"])

    if text and file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Run Fusion"):

            p1, prob1 = predict_text(bert_model, bert_tok, text)
            p2, prob2 = predict_text(rob_model, rob_tok, text)
            p3, prob3 = predict_image(img)

            results = [
                ("BERT", label_text[p1], prob1[p1] * 100),
                ("RoBERTa", label_text[p2], prob2[p2] * 100),
                ("ResNet", label_img[p3], prob3[p3] * 100)
            ]

            render_table(results)
            st.markdown(f"**Final Decision:** {final_decision(results)}")

st.markdown("---")
st.caption("Multimodal AI Detection System")