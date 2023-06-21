import jittor as jt
from jittor import nn
from jittor import Module
import numpy as np

class AlexNet(Module):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=8*8*64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def execute(self, x) -> None:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = x.view(-1, 8*8*64)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x