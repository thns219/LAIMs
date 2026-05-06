
I - Abstract
Hệ thống LAIMs (Large-scale AI-generated Multimodal Detection System) phát hiện nội dung đa phương tiện (văn bản + hình ảnh) do AI tạo ra sử dụng các mô hình học sâu hiện đại:

Văn bản: BERT (95.0% Accuracy), RoBERTa (89.97% Accuracy)
Hình ảnh: ResNet50 (75.11% Accuracy), Vision Transformer (52.4% Accuracy)
Multimodal Fusion: Weighted Late Fusion

II -  System Architecture
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Text Input    │───▶│ BERT/RoBERTa    │───▶│ Multimodal       │
│                 │    │ p_text ∈ [0,1]  │    │ Fusion           │
└─────────────────┘    └─────────────────┘    │ (Weighted Sum)   │
                                               │ p_final ∈ [0,1]  │──▶ Prediction
┌─────────────────┐    ┌─────────────────┐    │                  │
│  Image Input    │───▶│ ResNet50/ViT    │───▶│                  │
│                 │    │ p_image ∈ [0,1] │    └──────────────────┘
└─────────────────┘    └─────────────────┘

Fusion Formula:
p_final = 0.36×p_BERT + 0.34×p_RoBERTa + 0.20×p_ResNet + 0.10×p_ViT

III -  Quick Start
1. Clone Repository
git clone https://github.com/thns219/LAIMs.git
cd LAIMs
2. Install Dependencies
pip install -r requirements.txt
3. Download Pre-trained Models
# Models will be automatically downloaded by HuggingFace
python download_models.py
4. Run Demo
# Streamlit Web Demo
streamlit run app.py
# Command Line Detection
python detect.py --text "Your text here" --image "path/to/image.jpg"

IV - Key ReSults
<img width="764" height="241" alt="image" src="https://github.com/user-attachments/assets/500b1b2b-ecad-4f08-a6c1-94cc7bcfc2a0" />

<img width="895" height="521" alt="image" src="https://github.com/user-attachments/assets/56b78e31-7501-40b1-b12a-2b585930274e" />

Datasets
Text Dataset
Source: F3 Dataset [9]
Size: 10K train, 3K test
Classes: Human (0), AI-Generated (1)
Image Dataset
Source: Diffusion Data [10]
Size: ~3K images
Classes: Real (0), AI-Generated (1)
