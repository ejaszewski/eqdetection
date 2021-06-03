import torch
import torch.nn as nn

from eqdetection.layers.cnn import ResidualCNN
from eqdetection.layers.attention import Transformer
from eqdetection.layers.conditional import ConditionalAffineCoupling


class ConditioningNetwork(nn.Module):
    def __init__(self):
        super(ConditioningNetwork, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(nn.Conv1d(3, 16, 1))
        self.encoder.append(ResidualCNN(16, 5))
        self.encoder.append(ResidualCNN(16, 5))

        self.encoder.append(nn.MaxPool1d(5, 2, padding=2))
        self.encoder.append(nn.Conv1d(16, 32, 1))
        self.encoder.append(ResidualCNN(32, 5))
        self.encoder.append(ResidualCNN(32, 5))

        self.encoder.append(nn.MaxPool1d(5, 2, padding=2))
        self.encoder.append(nn.Conv1d(32, 64, 1))
        self.encoder.append(ResidualCNN(64, 5))
        self.encoder.append(ResidualCNN(64, 5))

        self.decoder.append(nn.Upsample(scale_factor=2))
        self.decoder.append(nn.Conv1d(64, 32, 1))
        self.decoder.append(ResidualCNN(32, 5))
        self.decoder.append(ResidualCNN(32, 5))

        # self.decoder.append(nn.Upsample(scale_factor=2))
        self.decoder.append(nn.Conv1d(32, 16, 1))
        self.decoder.append(ResidualCNN(16, 5))
        self.decoder.append(ResidualCNN(16, 5))

        self.decoder.append(nn.Conv1d(16, 4, 1))

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        for layer in self.decoder:
            x = layer(x)

        return x


class CouplingNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, latent):
        super(CouplingNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, latent, 5, padding=2)
        self.conv2 = nn.Conv1d(latent, latent, 1)
        self.conv3 = nn.Conv1d(latent, out_channels, 5, padding=2)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)

        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).tanh()

        return x


class ParallelModel(nn.Module):
    def __init__(self, blocks, length):
        super(ParallelModel, self).__init__()

        self.length = length

        self.cond = ConditioningNetwork()

        self.flow = nn.ModuleList()

        # NOTE: You can change the mask type by swapping the commented block.

        # Create a mask of alternating ones and zeros.
        mask = torch.arange(length) % 2

        # # Create a mask of zeros in one channel and ones in the other.
        # mask = torch.ones((2, length))
        # mask[1,:] = 0

        for _ in range(blocks):
            scale = CouplingNetwork(6, 2, 128)
            shift = CouplingNetwork(6, 2, 128)
            self.flow.append(ConditionalAffineCoupling(mask, scale, shift))

            # Update the mask so that it isn't the same for all blocks
            mask = mask.roll(1)
            # mask = 1 - mask

    def forward(self, x, y):
        y = self.cond(y)

        log_det = 0

        for flow in self.flow:
            x, layer_log_det = flow(x, y)
            log_det += layer_log_det

        return x, log_det

    def reverse(self, z, y):
        y = self.cond(y)

        log_det = 0

        for flow in reversed(self.flow):
            z, layer_log_det = flow.reverse(z, y)
            log_det += layer_log_det

        return z, log_det

    def infer(self, y, count, prior):
        y = self.cond(y).repeat(count, 1, 1)
        z = prior.sample((count, self.length)).transpose(1, 2).squeeze(dim=-1)

        for flow in reversed(self.flow):
            z, _ = flow.reverse(z, y)

        return z
