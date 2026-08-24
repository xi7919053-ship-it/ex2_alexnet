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

    # 自动选择当前能用的设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # 加载训练时保存的最佳模型
    model = AlexNet(num_classes=10)

    model.load_state_dict(
        torch.load(
            "weights/best.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    # 预测时的处理要和测试集保持一致
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # 待预测图片放在 images 文件夹里
    image_folder = "images"
    os.makedirs("outputs", exist_ok=True)

    # 一个列表存表格结果，另一个列表存画图需要的数据
    results = []
    display_images = []

    # 排序后每次显示的顺序都一样
    filenames = sorted(os.listdir(image_folder))

    # 逐张预测文件夹里的图片
    for filename in filenames:

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(image_folder, filename)

            # 文件名就是图片的真实类别，例如 cat.jpg 对应 cat
            true_class = os.path.splitext(filename)[0].lower()

            # 原图留着最后展示
            original_image = Image.open(image_path).convert("RGB")

            # 处理成模型需要的输入格式
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

            # 这一份最后写进 CSV
            results.append([
                filename,
                true_class,
                predicted_class,
                round(confidence, 2),
                correct
            ])

            # 这一份用来画预测结果图
            display_images.append({
                "filename": filename,
                "image": original_image,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "correct": correct
            })

    # 保存预测结果表
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

    # 把预测结果画出来
    num_images = len(display_images)

    if num_images > 0:
        cols = 2
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

        # 只有一行时 axes 的形状不一样，这里统一处理
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

        # 图片数量是奇数时，把最后多出的子图隐藏掉
        for j in range(num_images, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()

        figure_path = "outputs/predict_results.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Prediction figure saved to {figure_path}")


if __name__ == "__main__":
    main()
