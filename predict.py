import os
import torch
from PIL import Image
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

    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # 创建模型
    model = AlexNet(num_classes=10)

    # 加载训练好的最佳模型参数
    model.load_state_dict(
        torch.load(
            "weights/best.pth",
            map_location=device
        )
    )

    model = model.to(device)

    # 预测模式
    model.eval()

    # 与测试集相同的预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    image_folder = "images"

    for filename in os.listdir(image_folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                image_folder,
                filename
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            image = transform(image)

            # 增加 batch dimension
            # 3×32×32 -> 1×3×32×32
            image = image.unsqueeze(0)

            image = image.to(device)

            with torch.no_grad():

                outputs = model(image)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    1
                )

            predicted_class = classes[
                predicted.item()
            ]

            confidence = confidence.item() * 100

            print(
                f"{filename}: "
                f"{predicted_class} "
                f"({confidence:.2f}%)"
            )


if __name__ == "__main__":
    main()