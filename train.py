import time
import torch
import os


def train_model(
    model,
    trainloader,
    testloader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=2
):

    os.makedirs("weights", exist_ok=True)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []

    best_acc = 0.0

    for epoch in range(epochs):

        # =========================
        # 1. Training
        # =========================
        model.train()

        start_time = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            # 清空上一轮梯度
            optimizer.zero_grad()

            # forward
            outputs = model(images)

            # loss
            loss = criterion(outputs, labels)

            # backward
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

            # 找出预测类别
            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        # 一个 epoch 的平均 loss
        train_loss = running_loss / len(trainloader)

        # 一个 epoch 的训练准确率
        train_acc = 100.0 * correct / total

        # 一个 epoch 的训练时间
        epoch_time = time.time() - start_time


        # =========================
        # 2. Testing
        # =========================
        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for images, labels in testloader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = outputs.max(1)

                test_total += labels.size(0)

                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total


        # =========================
        # 3. Save results
        # =========================
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        epoch_times.append(epoch_time)


        # =========================
        # 4. Print
        # =========================
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Time: {epoch_time:.2f}s "
            f"| Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Test Acc: {test_acc:.2f}%"
        )


        # =========================
        # 5. Save best model
        # =========================
        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
                "weights/best.pth"
            )

        scheduler.step()

    
    # =========================
    # Average epoch time
    # =========================
    average_time = sum(epoch_times) / len(epoch_times)

    print("\nTraining finished!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Average Time per Epoch: {average_time:.2f}s")


    return (
        train_losses,
        train_accuracies,
        test_accuracies,
        epoch_times
    )
