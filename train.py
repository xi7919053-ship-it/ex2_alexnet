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
    epochs=200,
    start_epoch=0,
    best_acc=0.0,
    checkpoint_path="weights/checkpoint.pth",
):

    os.makedirs("weights", exist_ok=True)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(start_epoch, epochs):

        # 先训练一轮
        model.train()

        start_time = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            # 每个 batch 开始前先清空梯度
            optimizer.zero_grad()

            # 前向计算
            outputs = model(images)

            # 计算这批数据的损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新模型参数
            optimizer.step()

            running_loss += loss.item()

            # 取分数最高的类别
            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        # 算出这一轮的平均 loss
        train_loss = running_loss / len(trainloader)

        # 这一轮的训练准确率
        train_acc = 100.0 * correct / total

        # 记录这一轮用了多久
        epoch_time = time.time() - start_time


        # 再用测试集看一下效果
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


        # 记下这一轮的结果，后面用来画图
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        epoch_times.append(epoch_time)


        # 在控制台输出训练情况
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Time: {epoch_time:.2f}s "
            f"| LR: {current_lr:.5f} "
            f"| Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Test Acc: {test_acc:.2f}%"
        )


        # 测试准确率更高时保存当前模型
        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
            "weights/best.pth"
            )

        # 一轮结束后更新学习率
        scheduler.step()

        # 同时保存完整断点，下次可以接着跑
        torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
        },
        checkpoint_path
        )

    
    # 统计平均每轮训练时间
    if not epoch_times:
        print(f"Training already completed ({start_epoch}/{epochs} epochs).")
        return train_losses, train_accuracies, test_accuracies, epoch_times

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
