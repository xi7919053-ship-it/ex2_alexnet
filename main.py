"""
main.py —— AlexNet + CIFAR-10 训练入口
"""

import torch
import torch.nn as nn
import torch.optim as optim

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
    # 6. Training
    # =========================
    train_losses, train_accuracies, test_accuracies, epoch_times = train_model(
    model=model,
    trainloader=trainloader,
    testloader=testloader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=200
    )

if __name__ == "__main__":
    main()
