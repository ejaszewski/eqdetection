import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, in_channel, filters, bidirectional):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(in_channel, filters, bidirectional=bidirectional)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        return x
