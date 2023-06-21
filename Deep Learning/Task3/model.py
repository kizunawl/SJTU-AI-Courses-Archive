import jittor as jt
from jittor import Module
from jittor import nn

import matplotlib.pyplot as plt


class Net(Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int, batch_size:int) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.h = jt.zeros(1, input_size, hidden_size)
        self.c = jt.zeros(1, input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def execute(self, input):
        x = input
        lstm_out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        lstm_out = lstm_out[:, :, -1]
        x = self.fc(lstm_out)
        return x

    def clear_cells(self, isTraining=True, batch_size:int=-1):
        self.h = jt.zeros(1, self.input_size, self.hidden_size)
        self.c = jt.zeros(1, self.input_size, self.hidden_size)