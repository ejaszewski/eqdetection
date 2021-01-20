import torch
import torch.nn as nn
import numpy as np


class ActNorm1d(nn.Module):
    def __init__(self, channels):
        super(ActNorm1d, self).__init__()

        self.scale = nn.Parameter(torch.ones((1, channels, 1)))
        self.shift = nn.Parameter(torch.zeros((1, channels, 1)))

    def forward(self, x):
        x = self.scale * x + self.shift
        return x

class InvertibleConv1d(nn.Module):
    def __init__(self, channels):
        super(InvertibleConv1d, self).__init__()

        self.w = nn.Parameter(torch.empty(channels, channels, 1))
        nn.init.orthogonal_(self.w)
    
    def forward(self, x):
        x = nn.functional.conv1d(x, self.w)
        return x

class AffineCoupling1d(nn.Module):
    def __init__(self, inner):
        super(AffineCoupling1d, self).__init__()

        self.inner = inner

    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)
        
        log_s, t = self.inner(x_b).chunk(2, 1)
        s = torch.sigmoid(log_s)
        
        y_a = s * x_a + t
        y_b = x_b
        y = torch.cat((y_a, y_b), dim=1)

        return y

class GlowBlock(nn.Module):
    def __init__(self, channels, inner):
        super(GlowBlock, self).__init__()

        self.act_norm = ActNorm1d(channels)
        self.conv1x1 = InvertibleConv1d(channels)
        self.coupling = AffineCoupling1d(inner)

    def forward(self, x):
        x = self.act_norm(x)
        x = self.conv1x1(x)
        x = self.coupling(x)
        return x
