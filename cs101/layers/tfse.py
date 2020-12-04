import torch
import torch.nn as nn


class TimeFrequencySE(nn.Module):
    """
    Implementation of the time-frequency wise Squeeze-and-Excitation (tf-SE) 
    block described in:
    *Hu et al., Sound Event Detection in Multichannel Audio using Convolutional
    Time-Frequency-Channel Squeeze and Excitation, arXiv:1908.01399*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(TimeFrequencySE, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(num_channels, 1, 1)

    def forward(self, input_tensor, weights=None):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :param weights
        :return: output tensor
        """

        batch_size, num_channels, H = input_tensor.size()

        # The squeeze operation of the tf-SE is done using a 1-by-1 convolution 
        # which can be represented by the linear of all channels C at a location
        # (i, j)
        if weights:
            weights = weights.view(1, num_channels, 1, 1)
            out = self.conv(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        output_tensor = torch.mul(input_tensor, 
                        squeeze_tensor.expand_as(input_tensor))


        return output_tensor