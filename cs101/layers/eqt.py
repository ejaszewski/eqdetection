import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, channels, units, activation=nn.ReLU, dropout=0.0):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(channels, units)
        self.linear2 = nn.Linear(units, channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, seq_len, channels, units, width=None):
        super(Attention, self).__init__()

        self.Wt = nn.Linear(channels, units, bias=False)
        self.Wx = nn.Linear(channels, units, bias=False)
        self.Wa = nn.Linear(units, 1)

        self.bh = nn.Parameter(torch.zeros(units))

        self.tanh = nn.Tanh()

        self.seq_len = seq_len
        self.width = width

    def __emission(self, x):
        q = torch.unsqueeze(self.Wt(x), 2)
        k = torch.unsqueeze(self.Wx(x), 1)
        h = self.tanh(q + k + self.bh)

        e = torch.reshape(self.Wa(h), (-1, self.seq_len, self.seq_len))

        return e

    def forward(self, x):
        x = x.transpose(1, 2)

        e = self.__emission(x)
        e = torch.exp(e - torch.max(e, -1, keepdim=True)[0])

        if self.width != None:
            lower = torch.arange(0, self.seq_len).to(
                x.device) - (self.width // 2)
            lower = lower.unsqueeze(-1)
            upper = lower + self.width
            indices = torch.arange(0, self.seq_len).to(x.device).unsqueeze(0)
            e = e * (lower <= indices).float() * (indices < upper).float()

        s = torch.sum(e, -1, keepdim=True)
        a = e / (s + 1e-7)

        v = torch.matmul(a, x).transpose(1, 2)

        return v, a


class Transformer(nn.Module):
    def __init__(self, seq_len, channels, dropout=0.1):
        super(Transformer, self).__init__()

        self.attention = Attention(seq_len, channels, 32)
        self.ff = FeedForward(channels, 128, dropout=dropout)
        self.norm1 = nn.LayerNorm((channels, seq_len))
        self.norm2 = nn.LayerNorm((channels, seq_len))

    def forward(self, x):
        att, w = self.attention(x)
        att = x + att
        norm = self.norm1(att)

        ff = self.ff(norm)
        ff = ff + norm

        out = self.norm2(ff)

        return out, w
