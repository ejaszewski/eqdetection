import torch
import torch.nn as nn

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

class InvertibleSigmoid(nn.Module):
    def __init__(self):
        super(InvertibleSigmoid, self).__init__()

    def _f(x):
        return 1 / (1 + (-x).exp())
    
    def _finv(x):
        return -(1 / x - 1).log()

    def _df(x):
        enx = (-x).exp()
        return enx / (enx + 1).square()

    def _dfinv(x):
        return 1 / (x - x.square())
    
    def forward(self, x):
        return InvertibleSigmoid._f(x), InvertibleSigmoid._df(x).sum(1)

    def reverse(self, z):
        return InvertibleSigmoid._finv(z), InvertibleSigmoid._dfinv(z).sum(1)

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, mask):
        super(ConditionalAffineCoupling, self).__init__()
        
        self.s = CouplingNetwork(6, 1, 128)
        self.t = CouplingNetwork(6, 1, 128)
        
        self.mask = nn.Parameter(mask, requires_grad=False)
    
    def forward(self, x, y):
        masked = self.mask * x
        
        s = self.s(masked, y)
        t = self.t(masked, y)
        z = masked + (1 - self.mask) * (x * s.exp() + t)
        
        log_det = (s * (1 - self.mask)).sum(1)

        return z, log_det
    
    def reverse(self, z, y):
        masked = self.mask * z
        
        s = self.s(masked, y)
        t = self.t(masked, y)
        x = masked + (1 - self.mask) * ((z - t) * (-s).exp())

        return x, (-s * (1 - self.mask)).sum(1)

class ResCNN(nn.Module):
    def __init__(self, channels, kernel, activation=nn.ReLU):
        super(ResCNN, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(channels)
        self.activation1 = activation()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=int(kernel / 2))

        self.batch_norm2 = nn.BatchNorm1d(channels)
        self.activation2 = activation()
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=int(kernel / 2))

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

        return x + residual


class ConditioningNetwork(nn.Module):
    def __init__(self):
        super(ConditioningNetwork, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(nn.Conv1d(3, 16, 1))
        self.encoder.append(ResCNN(16, 5))
        self.encoder.append(ResCNN(16, 5))

        self.encoder.append(nn.MaxPool1d(5, 2, padding=2))
        self.encoder.append(nn.Conv1d(16, 32, 1))
        self.encoder.append(ResCNN(32, 5))
        self.encoder.append(ResCNN(32, 5))
        
        self.encoder.append(nn.MaxPool1d(5, 2, padding=2))
        self.encoder.append(nn.Conv1d(32, 64, 1))
        self.encoder.append(ResCNN(64, 5))
        self.encoder.append(ResCNN(64, 5))

        self.decoder.append(nn.Upsample(scale_factor=2))
        self.decoder.append(nn.Conv1d(64, 32, 1))
        self.decoder.append(ResCNN(32, 5))
        self.decoder.append(ResCNN(32, 5))

        self.decoder.append(nn.Upsample(scale_factor=2))
        self.decoder.append(nn.Conv1d(32, 16, 1))
        self.decoder.append(ResCNN(16, 5))
        self.decoder.append(ResCNN(16, 5))

        self.decoder.append(nn.Conv1d(16, 4, 1))
    
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        
        for layer in self.decoder:
            x = layer(x)

        return x

class ParallelModel(nn.Module):
    def __init__(self, blocks, length, sigmoid=False):
        super(ParallelModel, self).__init__()

        self.cond = ConditioningNetwork()

        if sigmoid:
            self.sig = InvertibleSigmoid()
        else:
            self.sig = None

        self.flow = nn.ModuleList()

        mask = torch.arange(2 * length).reshape(2, length) % 2
        
        for _ in range(blocks):
            self.flow.append(ConditionalAffineCoupling(mask))
            mask = mask.roll(1)
    
    def forward(self, x, y):
        y = self.cond(y)

        if self.sig is not None:
            x, log_det = self.sig.reverse(x)
        else:
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
        
        if self.sig is not None:
            z, sig_log_det = self.sig(z)
            log_det += sig_log_det

        return z, log_det

    def infer(self, y, count, prior):
        y = self.cond(y).repeat(count, 1, 1)
        z = prior.sample((count, self.length)).transpose(1, 2).squeeze(dim=-1)

        for flow in reversed(self.flow):
            z, _ = flow.reverse(z, y)

        if self.sig is not None:
            z, _ = self.sig(z)
        
        return z