
I - Abstract
Hệ thống LAIMs (Large-scale AI-generated Multimodal Detection System) phát hiện nội dung đa phương tiện (văn bản + hình ảnh) do AI tạo ra sử dụng các mô hình học sâu hiện đại:

Văn bản: BERT (95.0% Accuracy), RoBERTa (89.97% Accuracy)
Hình ảnh: ResNet50 (75.11% Accuracy), Vision Transformer (52.4% Accuracy)
Multimodal Fusion: Weighted Late Fusion

II -  System Architecture

<img width="711" height="235" alt="image" src="https://github.com/user-attachments/assets/71d669d4-1442-4c8a-96ec-eb4cdb7bdc77" />

Fusion Formula:
p_final = 0.36×p_BERT + 0.34×p_RoBERTa + 0.20×p_ResNet + 0.10×p_ViT

III -  Quick Start
# ⚙️ Cài đặt và chạy dự án

Đầu tiên clone repository về máy bằng lệnh:

git clone https://github.com/thns219/LAIMs.git
cd LAIMs

Tiếp theo tạo môi trường ảo để cài đặt thư viện.

Đối với Windows:

python -m venv venv
venv\Scripts\activate

Đối với Linux hoặc MacOS:

python3 -m venv venv
source venv/bin/activate

Sau khi kích hoạt môi trường ảo, tiến hành cài đặt các thư viện cần thiết:

pip install -U pip
pip install -r requirements.txt

Hoặc cài đặt thủ công:

pip install torch torchvision transformers scikit-learn pandas numpy matplotlib opencv-python pillow streamlit tqdm

Chuẩn bị dữ liệu văn bản bằng cách tải dataset F3 tại: https://github.com/mickeymst/F3/tree/main/F3_Dataset

Sau đó đặt file final_dataset.csv vào thư mục: data

Tiếp theo tải dataset hình ảnh: git clone https://github.com/thns219/diffusion_data.git

Đưa dữ liệu vào thư mục: diffusion_data
Cấu trúc dữ liệu ảnh:

 diffusion_data
├── 0_real/
└── 1_fake/
└── test/
└── train/
└── val
Huấn luyện các mô hình:

Train BERT: python src/train_bert.py

Train RoBERTa: python src/train_roberta.py

Train ResNet50: python src/image_ai/train_resnet.py

Train Vision Transformer (ViT): python src/image_ai/train.py

Chạy giao diện demo Streamlit: streamlit run app/streamlit_app.py

Yêu cầu hệ thống:

Python >= 3.9
RAM >= 8GB
GPU NVIDIA hỗ trợ CUDA (khuyến nghị)
PyTorch >= 2.0

Các công nghệ sử dụng trong dự án:
PyTorch
HuggingFace Transformers
Scikit-learn
OpenCV
Pillow
Streamlit
4. Run Demo
# Streamlit Web Demo
streamlit run app.py
<img width="459" height="870" alt="Ảnh chụp màn hình 2026-04-19 164232" src="https://github.com/user-attachments/assets/b5dc0335-bc18-457f-883d-7f5d8333c6c2" />

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
