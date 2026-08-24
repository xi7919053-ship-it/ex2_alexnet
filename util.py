import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_history(csv_path="outputs/training_history.csv"):

    # 从 CSV 读取每一轮的训练结果
    history = pd.read_csv(csv_path)

    os.makedirs("outputs", exist_ok=True)

    # 画训练损失曲线
    plt.figure()

    plt.plot(
        history["Epoch"],
        history["Train Loss"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.savefig("outputs/loss_curve.png")
    plt.close()


    # 画训练集和测试集准确率曲线
    plt.figure()

    plt.plot(
        history["Epoch"],
        history["Train Accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["Epoch"],
        history["Test Accuracy"],
        label="Test Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")

    plt.legend()

    plt.savefig("outputs/accuracy_curve.png")
    plt.close()

    print("Curves saved to outputs/")


if __name__ == "__main__":
    plot_history()
