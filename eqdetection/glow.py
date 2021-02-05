import torch.nn as nn

from eqdetection.layers.flows import ActNorm1d, InvertibleConv1d, AffineCoupling1d, GlowBlock


class AffineInner(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(AffineInner, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel,
            padding=int(kernel / 2))

        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels,
            in_channels,
            1)

        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            padding=int(kernel / 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.activation2(x)

        x = self.conv3(x)

        return x


class GlowFlow(nn.Module):
    def __init__(self, channels, blocks):
        super(GlowFlow, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(
                GlowBlock(channels, AffineInner(channels // 2, channels, 3)))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampler, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.activation2 = nn.ReLU()

        self.pool = nn.MaxPool1d(3, 2, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.activation2(x)

        x = self.pool(x)

        return x

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampler, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.activation2 = nn.ReLU()
    
    
    def forward(self, x):
        x = self.upsample(x)
        
        x = self.conv1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.activation2(x)

        return x


class GlowNetwork(nn.Module):
    def __init__(self, in_channels, levels, blocks):
        super(GlowNetwork, self).__init__()

        self.levels = levels

        self.dsample = nn.ModuleList()
        self.flows = nn.ModuleList()
        self.decoder_p = nn.ModuleList()
        self.decoder_s = nn.ModuleList()
        self.decoder_e = nn.ModuleList()

        channels = in_channels
        for _ in range(levels):
            out_channels = int(channels * 1.5)
            out_channels -= out_channels % 2

            self.dsample.append(Downsampler(channels, out_channels))
            self.flows.append(GlowFlow(out_channels, blocks))
            self.decoder_p.append(Upsampler(out_channels, channels))
            self.decoder_s.append(Upsampler(out_channels, channels))
            self.decoder_e.append(Upsampler(out_channels, channels))

            channels = out_channels
        
        self.decoder_p.append(nn.Conv1d(3, 1, 3, 1, 1))
        self.decoder_s.append(nn.Conv1d(3, 1, 3, 1, 1))
        self.decoder_e.append(nn.Conv1d(3, 1, 3, 1, 1))

    def level_forward(self, i, x):
        x = self.dsample[i](x)
        x = self.flows[i](x)
        
        p = x
        s = x
        e = x

        if i < self.levels - 1:
            p_n, s_n, e_n = self.level_forward(i + 1, x)
            p = p + p_n
            s = s + s_n
            e = e + e_n

        p = self.decoder_p[i](p)
        s = self.decoder_s[i](s)
        e = self.decoder_e[i](e)
        
        return p, s, e

    def forward(self, x):
        p, s, e = self.level_forward(0, x)
        p = self.decoder_p[-1](p)
        s = self.decoder_s[-1](s)
        e = self.decoder_e[-1](e)
        return p, s, e
