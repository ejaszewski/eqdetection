import torch.nn as nn


class PoolingCNN(nn.Module):
    """ A simple CNN that applies a pooling layer.

    This class implements a simple block that combines a convolutional layer
    with a pooling layer. The convolution is applied first, followed by the
    pooling, and optionally, the activation.

    Attributes:
        conv: The convolutional layer.
        pooling: The pooling layer.
        activation: The activation layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 pool_size,
                 activation=nn.ReLU):
        """Initializes a PoolingCNN with the given parameters.

        For information on the in_channels, out_channels, kernel_size, and
        padding parameters, see the documentation for torch.nn.Conv1d. For
        information on the pool_size parameter, see the documentation for
        torch.nn.MaxPool1d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: The kernel size of the convolutional layer.
            padding: The padding applied to the convolutional layer.
            pool_size: The size of the pooling applied.
            activation: Optional; The activation to apply.
                Default: torch.nn.ReLU.
        """

        super(PoolingCNN, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=padding)
        self.pool = nn.MaxPool1d(pool_size)
        self.activation = activation()

    def forward(self, x):
        """Performs a forward pass of the block.

        Args:
            x: Input to the block.
        
        Returns:
            Output of the block.
        """

        x = self.conv(x)
        x = self.pool(x)
        x = self.activation(x)
        return x


class UpsampleCNN(nn.Module):
    """ A simple CNN that applies an upsampling layer.

    This class implements a simple block that combines a convolutional layer
    with an upsampling layer. The upsampling is applied first, followed by the
    convolution, and optionally, the activation.

    Attributes:
        conv: The convolutional layer.
        upsample: The upsampling layer.
        activation: The activation layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 scale,
                 activation=nn.ReLU):
        """Initializes an UpsampleCNN with the given parameters.

        For information on the in_channels, out_channels, kernel_size, and
        padding parameters, see the documentation for torch.nn.Conv1d. For
        information on the scale parameter, see the documentation for
        torch.nn.Upsample.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: The kernel size of the convolutional layer.
            padding: The padding applied to the convolutional layer.
            scale: The scale factor used for the upsample.
            activation: Optional; The activation to apply.
                Default: torch.nn.ReLU.
        """

        super(UpsampleCNN, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale)
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=padding)

        if activation != None:
            self.activation = activation()

    def forward(self, x):
        """Performs a forward pass of the block.

        Args:
            x: Input to the block.
        
        Returns:
            Output of the block.
        """

        x = self.upsample(x)
        x = self.conv(x)
        if self.activation != None:
            x = self.activation(x)
        return x


class ResidualCNN(nn.Module):
    """A simple two-layer CNN with residual connection.

    This class implements a simple residual CNN with two layers. Each of the
    two convolutional layers is implemented as:
        CNN -> Activation -> BatchNorm -> Dropout
    And the residual connection is applied by adding the input to the output
    of the second convolutional layer.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        activation1: First layer activation.
        activation2: Second layer activation.
        batch_norm1: First layer batch norm.
        batch_norm2: Second layer batch norm.
        dropout: First layer dropout.
        dropout: Second layer dropout.
    """
    def __init__(self, channels, kernel, dropout=0.0, activation=nn.ReLU):
        """Initializes a ResidualCNN block with the given parameters.

        Note: Padding of size kernel / 2 is implicitly added to keep output
        shape the same as input shape.

        Args:
            channels: Number of channels for the input and output.
            kernel: The size of the kernel for both convolutions.
            dropout: Optional; The dropout probability, applied to both
                layers. Default: 0
            activation: Optional; The activation applied to both layers.
                Default: torch.nn.ReLU
        """

        super(ResidualCNN, self).__init__()

        self.conv1 = nn.Conv1d(channels,
                               channels,
                               kernel,
                               padding=int(kernel / 2))
        self.conv2 = nn.Conv1d(channels,
                               channels,
                               kernel,
                               padding=int(kernel / 2))

        self.batch_norm1 = nn.BatchNorm1d(channels)
        self.batch_norm2 = nn.BatchNorm1d(channels)

        self.activation1 = activation()
        self.activation2 = activation()

        self.dropout1 = nn.Dropout2d(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)

    def forward(self, x):
        """Performs a forward pass of the block.

        Args:
            x: Input to the block.
        
        Returns:
            Output of the block.
        """

        residual = x

        x = self.conv1(x)
        x = self.activation1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        return x + residual
