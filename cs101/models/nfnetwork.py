import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

from cs101.layers.cnn import PoolingCNN, ResidualCNN, UpsampleCNN
from cs101.layers.rnn import SimpleLSTM
from cs101.layers.eqt import Transformer, Attention
from cs101.layers.flows import ActNorm

class NFEncoder(nn.Module):
    def __init__(self, in_channels, conv, res_cnn, bilstm):
        super(NFEncoder, self).__init__()

        self.cnn = nn.ModuleList()
        self.res = nn.ModuleList()
        self.bilstm = nn.ModuleList()
        self.flows = nn.ModuleList()

        prev_channels = in_channels
        
        self.flows.append(ActNorm(dim = [1000,3,6000]))

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

        # Add the BiLSTM layers
        for _ in range(bilstm):
            self.bilstm.append(SimpleLSTM(prev_channels, 16, True))
            self.bilstm.append(nn.Conv1d(32, 16, 1))
            self.bilstm.append(nn.BatchNorm1d(16))
            prev_channels = 16

        self.lstm = SimpleLSTM(prev_channels, 16, False)

    def forward(self, x):
        m, _, _ = x.shape
        if m == 1000:
            log_det = torch.zeros(m)
            for flow in self.flows:
                x, ld = flow.forward(x)
        
        # Run x through the CNNs
        for layer in self.cnn:
            x = layer(x)

        # Run x through the ResCNNs
        for layer in self.res:
            x = layer(x)

        # Run x through the BiLSTMs
        for layer in self.bilstm:
            x = layer(x)

        # LSTM Layer
        x = self.lstm(x)

        return x
    
class NFDecoder(nn.Module):
    def __init__(self, in_channels, conv, last, lstm=False, attention=False):
        super(NFDecoder, self).__init__()

        # Add LSTM (if requested)
        if lstm:
            self.lstm = SimpleLSTM(in_channels, in_channels, False)
        else:
            self.lstm = None

        # Add Attention (if requested)
        if attention:
            self.attention = Attention(46, in_channels, 32, width=3)
        else:
            self.attention = None

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
        # Run x through the LSTM if it exists
        if self.lstm is not None:
            x = self.lstm(x)

        # Run x through the attention if it exists
        if self.attention is not None:
            x, _ = self.attention(x)

        # Run x through the CNNs
        for layer in self.cnn:
            x = layer(x)

        return x
    
class NFNetwork(nn.Module):
    def __init__(
        self,
        conv_ds=[(8, 11, 5), (16, 9, 4), (16, 7, 3), (32, 7, 3),
                 (32, 5, 2), (64, 5, 2), (64, 3, 1)],
        res_cnn=[(64, 3), (64, 3), (64, 3), (64, 3), (64, 3)],
        conv_us=[(96, 3, 1), (96, 5, 3), (32, 5, 3), (32, 7, 4),
                 (16, 7, 3), (16, 9, 4), (8, 11, 5)],
        bilstm=2
    ):
        super(NFNetwork, self).__init__()

        # Universal encoder
        self.encoder = NFEncoder(3, conv_ds, res_cnn, bilstm)

        # Transformers
        self.transformer1 = Transformer(46, 16)
        self.transformer2 = Transformer(46, 16)

        # Separate decoders
        self.decoder_p = NFDecoder(
            16, conv_us, (1, 11, 5), lstm=True, attention=True)
        self.decoder_s = NFDecoder(
            16, conv_us, (1, 11, 5), lstm=True, attention=True)
        self.decoder_c = NFDecoder(16, conv_us, (1, 11, 5))

    def forward(self, x):
        # Combined encoder
        x = self.encoder(x)

        # Transformers
        x, _ = self.transformer1(x)
        x, _ = self.transformer2(x)

        # Separate decoders
        p = self.decoder_p(x)
        s = self.decoder_s(x)
        c = self.decoder_c(x)

        return p, s, c
