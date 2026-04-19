import os
import csv

# ====== TỰ ĐỘNG LẤY ĐƯỜNG DẪN ======
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "diffusion_data")

print("Đường dẫn dataset:", dataset_path)

# ====== CẤU HÌNH ======
classes = {
    "0_real": 0,
    "1_fake": 1
}

valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

output_csv = os.path.join(base_dir, "diffusion_data/dataset.csv")

total = 0

# ====== GHI CSV ======
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])

    for folder_name, label in classes.items():
        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.exists(folder_path):
            print("❌ Không tìm thấy:", folder_path)
            continue

        print(f"Đang xử lý: {folder_name}")

        for root, _, files in os.walk(folder_path):  # đọc cả folder con
            for file in files:
                if file.lower().endswith(valid_exts):
                    full_path = os.path.join(root, file)

                    # Chuẩn hóa path (tránh lỗi \ trên Windows)
                    full_path = full_path.replace("\\", "/")

                    writer.writerow([full_path, label])
                    total += 1

print("\nDONE!")
print("Tổng số ảnh:", total)
print("File CSV:", output_csv)