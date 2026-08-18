import torch
import torch.nn as nn



class AlexNet(nn.Module):
    
    def __init__(self, num_classes=10):

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(

            # Conv1: 1x3x32x32 ->1x64x16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # MaxPool: 1x64x16x16 -> 1x64x8x8
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2: 1x64x8x8 -> 1x192x8x8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # MaxPool: 1x192x8x8 -> 1x192x4x4
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3: 1x192x4x4 ->1x384x4x4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 1x384x4x4 -> 1x256x4x4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 1x256x4x4-> 1x256x4x4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # MaxPool: 1x256x4x4 -> 1x256x2x2
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.5),

            # Flatten: 256x2x2=1024
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