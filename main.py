"""
main.py —— AlexNet + CIFAR-10 训练入口
"""

import torch
import torch.nn as nn
import torch.optim as optim

import os
import csv
import matplotlib.pyplot as plt

from data import get_dataloaders
from models.alexnet import AlexNet
from train import train_model


def main():

    # =========================
    # 1. Device
    # =========================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")

    else:
        device = torch.device("cpu")
        print("Using CPU")


    # =========================
    # 2. Data
    # =========================
    trainloader, testloader = get_dataloaders(
        batch_size=128
    )


    # =========================
    # 3. Model
    # =========================
    model = AlexNet(num_classes=10)

    model = model.to(device)


    # =========================
    # 4. Loss
    # =========================
    criterion = nn.CrossEntropyLoss()


    # =========================
    # 5. Optimizer
    # =========================
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[80, 140, 180],
    gamma=0.1
    )

    # =========================
    # 6. Load checkpoint
    # =========================
    checkpoint_dir = os.environ.get("ALEXNET_CHECKPOINT_DIR", "weights")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "checkpoint.pth"
    )

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("best_acc", 0.0)

        print(
            f"Checkpoint loaded. "
            f"Resume training from epoch {start_epoch + 1}"
        )

    else:
        print("No checkpoint found. Start training from epoch 1.")


    # =========================
    # 7. Training
    # =========================


    train_losses, train_accuracies, test_accuracies, epoch_times = train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=200,
        start_epoch=start_epoch,
        best_acc=best_acc,
        checkpoint_path=checkpoint_path
    )

    # =========================
    # 8. Save results
    # =========================
    os.makedirs("outputs", exist_ok=True)

    # Save training history
    with open("outputs/training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Train Accuracy",
            "Test Accuracy",
            "Time"
        ])

        for i in range(len(train_losses)):
            writer.writerow([
                i + 1,
                train_losses[i],
                train_accuracies[i],
                test_accuracies[i],
                epoch_times[i]
            ])

    # Loss curve
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("outputs/loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.savefig("outputs/accuracy_curve.png")
    plt.close()

    print("Training results saved to outputs/")

if __name__ == "__main__":
    main()
