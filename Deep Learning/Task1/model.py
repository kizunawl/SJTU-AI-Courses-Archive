import jittor as jt
from jittor import Module
from jittor import nn

class Model(Module):
    def __init__(self) -> None:
        super().__init__(Module)
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x