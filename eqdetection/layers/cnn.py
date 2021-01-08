import torch
import torch.nn as nn


class PoolingCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, activation=nn.ReLU):
        super(PoolingCNN, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding)
        self.pool = nn.MaxPool1d(pool_size)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.activation(x)
        return x


class UpsampleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, scale, activation=nn.ReLU):
        super(UpsampleCNN, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding)

        if activation != None:
            self.activation = activation()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        if self.activation != None:
            x = self.activation(x)
        return x


class ResidualCNN(nn.Module):
    def __init__(self, channels, kernel, dropout, activation=nn.ReLU):
        super(ResidualCNN, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(channels)
        self.activation1 = activation()
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=int(kernel / 2))

        self.batch_norm2 = nn.BatchNorm1d(channels)
        self.activation2 = activation()
        self.dropout2 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=int(kernel / 2))

    def forward(self, x):
        residual = x
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv1(x)

        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.conv2(x)

        return x + residual
