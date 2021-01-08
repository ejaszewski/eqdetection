import torch.nn as nn

from eqdetection.layers.cnn import PoolingCNN, ResidualCNN, UpsampleCNN
from eqdetection.layers.attention import Transformer, Attention


class Encoder(nn.Module):
    def __init__(self, in_channels, conv, res_cnn):
        super(Encoder, self).__init__()

        self.cnn = nn.ModuleList()
        self.res = nn.ModuleList()

        prev_channels = in_channels

        # Add all of the convolutional layers
        for channels, kernel, padding in conv:
            self.cnn.append(
                PoolingCNN(prev_channels, channels, kernel, padding, 2))
            prev_channels = channels

        # Add all of the ResCNN layers
        for (channels, kernel) in res_cnn:
            self.res.append(
                ResidualCNN(channels, kernel, 0.1))
            prev_channels = channels

    def forward(self, x):
        # Run x through the CNNs
        for layer in self.cnn:
            x = layer(x)

        # Run x through the ResCNNs
        for layer in self.res:
            x = layer(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, conv, last):
        super(Decoder, self).__init__()

        prev_channels = in_channels

        self.cnn = nn.ModuleList()

        # Add convolutional layers
        for channels, kernel, padding in conv:
            self.cnn.append(
                UpsampleCNN(prev_channels, channels, kernel, padding, 2))
            prev_channels = channels

        # Add activation-less final layer
        self.cnn.append(
            nn.Conv1d(prev_channels, last[0], last[1], padding=last[2]))

    def forward(self, x):
        # Run x through the CNNs
        for layer in self.cnn:
            x = layer(x)

        return x


class Network(nn.Module):
    def __init__(
        self,
        conv_ds=[(8, 11, 5), (16, 9, 4), (16, 7, 3), (32, 7, 3),
                 (32, 5, 2), (64, 5, 2), (64, 3, 1)],
        res_cnn=[(64, 3), (64, 3), (64, 3), (64, 3), (64, 3)],
        conv_us=[(96, 3, 1), (96, 5, 3), (32, 5, 3), (32, 7, 4),
                 (16, 7, 3), (16, 9, 4), (8, 11, 5)],
    ):
        super(Network, self).__init__()

        # Universal encoder
        self.encoder = Encoder(3, conv_ds, res_cnn)

        # Transformers
        self.transformer1 = Transformer(46, 64)
        self.transformer2 = Transformer(46, 64)

        # Separate decoders
        self.decoder_p = Decoder(64, conv_us, (1, 11, 5))
        self.decoder_s = Decoder(64, conv_us, (1, 11, 5))
        self.decoder_e = Decoder(64, conv_us, (1, 11, 5))

    def forward(self, x):
        # Combined encoder
        x = self.encoder(x)

        # Transformers
        x, _ = self.transformer1(x)
        x, _ = self.transformer2(x)

        # Separate decoders
        p = self.decoder_p(x)
        s = self.decoder_s(x)
        e = self.decoder_e(x)

        return p, s, e
