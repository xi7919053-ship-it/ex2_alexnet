import torch
import torch.nn as nn



class AlexNet(nn.Module):
    
    def __init__(self, num_classes=10):

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(

            # 第一层先把尺寸降到 16×16
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 池化后变成 64×8×8
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层增加通道数，图片大小不变
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 再池化到 192×4×4
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 后面三层卷积继续提取特征
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 通道数调整为 256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 最后一层卷积保持尺寸不变
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 最终得到 256×2×2 的特征图
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.5),

            # 展开后共有 1024 个特征
            nn.Linear(256 * 2 * 2, 4096), 

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)

        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = AlexNet()

    x = torch.randn(1, 3, 32, 32)

    y = model(x)

    print(y.shape)
