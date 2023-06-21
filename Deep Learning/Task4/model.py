import jittor as jt
from jittor import Module
from jittor import nn 
import pygmtools as pygm
import numpy as np
import parameter


class AlexNet(Module):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.padsize = parameter.parameters().pad
        self.kernel_size = parameter.parameters().kernel_size
        self.side_len = parameter.parameters().side_len

        self.CNNoutSize = int(self.side_len / 2 + self.padsize - (self.kernel_size - 1) / 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=self.padsize),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.CNNoutSize*self.CNNoutSize*32, out_features=128),
            nn.Relu(),
            nn.Linear(in_features=128, out_features=16)
        )

    def execute(self, x) -> None:
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = x.view(-1, self.CNNoutSize*self.CNNoutSize*32)
        x = self.fc(x)
        return x


class Net(Module):
    def __init__(self, sinkhorn_norm) -> None:
        pygm.BACKEND = 'jittor'
        self.slice = parameter.parameters().slice**2
        self.side_len = parameter.parameters().side_len

        if (sinkhorn_norm):
            self.execute = self.execute_sinkhorn
        else:
            self.execute = self.execute_sigmoid

        self.AlexNet = AlexNet()
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.slice*16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.slice ** 2),
        )
    
    def execute_sinkhorn(self, input) -> None:
        x = input
        x = jt.reshape(x, (-1, 3, self.side_len, self.side_len))
        x = self.AlexNet(x)
        x = jt.reshape(x, (-1, self.slice*16))
        x = self.fc(x)
        x = jt.reshape(x, (-1, self.slice, self.slice))
        x = pygm.sinkhorn(x)
        return x
    
    def execute_sigmoid(self, input) -> None:
        x = input
        x = jt.reshape(x, (-1, 3, self.side_len, self.side_len))
        x = self.AlexNet(x)
        x = jt.reshape(x, (-1, self.slice*16))
        x = self.fc(x)
        x = jt.nn.Sigmoid()(x)
        x = jt.reshape(x, (-1, self.slice, self.slice))
        return x
