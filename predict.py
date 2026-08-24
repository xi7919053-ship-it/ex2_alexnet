import os
import csv
import math
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.alexnet import AlexNet


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def main():

    # =========================
    # 1. Device
    # =========================
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # =========================
    # 2. Load model
    # =========================
    model = AlexNet(num_classes=10)

    model.load_state_dict(
        torch.load(
            "weights/best.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    # =========================
    # 3. Transform
    # =========================
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # =========================
    # 4. Image folder
    # =========================
    image_folder = "images"
    os.makedirs("outputs", exist_ok=True)

    # 保存结果
    results = []
    display_images = []

    # 为了显示顺序固定，排序一下
    filenames = sorted(os.listdir(image_folder))

    # =========================
    # 5. Predict images
    # =========================
    for filename in filenames:

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(image_folder, filename)

            # 从文件名提取真实类别
            # 例如 cat.jpg -> cat
            true_class = os.path.splitext(filename)[0].lower()

            # 原始图像（用于显示）
            original_image = Image.open(image_path).convert("RGB")

            # 模型输入图像
            image = transform(original_image)
            image = image.unsqueeze(0)   # 3x32x32 -> 1x3x32x32
            image = image.to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = classes[predicted.item()]
            confidence = confidence.item() * 100
            correct = (true_class == predicted_class)

            print(
                f"{filename}: "
                f"True = {true_class}, "
                f"Predicted = {predicted_class}, "
                f"Confidence = {confidence:.2f}%, "
                f"Correct = {correct}"
            )

            # 保存到 CSV 的结果
            results.append([
                filename,
                true_class,
                predicted_class,
                round(confidence, 2),
                correct
            ])

            # 保存到绘图列表
            display_images.append({
                "filename": filename,
                "image": original_image,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "correct": correct
            })

    # =========================
    # 6. Save CSV
    # =========================
    csv_path = "outputs/predict_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Image",
            "True Class",
            "Predicted Class",
            "Confidence (%)",
            "Correct"
        ])

        writer.writerows(results)

    print(f"\nPrediction results saved to {csv_path}")

    # =========================
    # 7. Save result figure
    # =========================
    num_images = len(display_images)

    if num_images > 0:
        cols = 2
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

        # 如果只有一行或一列，统一转成可迭代形式
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        axes_flat = []
        for row in axes:
            for ax in row:
                axes_flat.append(ax)

        for i, item in enumerate(display_images):
            ax = axes_flat[i]
            ax.imshow(item["image"])
            ax.axis("off")

            ax.set_title(
                f"Image: {item['filename']}\n"
                f"True: {item['true_class']}\n"
                f"Pred: {item['predicted_class']}\n"
                f"Conf: {item['confidence']:.2f}%\n"
                f"Correct: {item['correct']}",
                fontsize=10
            )

        # 多余的子图去掉
        for j in range(num_images, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()

        figure_path = "outputs/predict_results.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Prediction figure saved to {figure_path}")


if __name__ == "__main__":
    main()