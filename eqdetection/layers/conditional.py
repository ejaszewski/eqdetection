from torch import nn


class ConditionalAffineCoupling(nn.Module):
    """A conditional affine coupling block.

    This class implements a conditional version of the masked affine coupling
    block described in Dinh et al. 2017 (arXiv:1605.08803). The affine
    coupling block is made conditional by passing a secondary conditioning
    input to the scale and shift networks in both the forward and reverse
    directions. Masking is applied only to the flow inputs.

    Note that the "flow" input is the input belonging to the flow itself (i.e.,
    it is the one being transformed), while the "conditioning" input is the
    input which is conditioned over by the scale and shift networks.

    Attributes:
        s: The scaling network.
        t: The shifting network.
        mask: The input mask.
    """
    def __init__(self, mask, scale, shift):
        """ Initializes ConditionalAffineCoupling with the given parameters.

        This function initializes a ConditionalAffineCoupling block with a
        given mask, scaling, and shifting network. Note that there are certain
        (unenforced) requirements in order for this block to work. Notably:
            - The scale and shift networks must accept two torch.Tensors as
              input (flow and conditioning inputs respectively).
            - The scale and shift networks must have an output whose shape is
              the same as the flow input.
            - The mask must be a binary torch.Tensor with the same shape as
              the flow input.

        Args:
            mask: The input mask to use.
            scale: The scaling network to use.
            shift: The shifting network to use.
        """

        super(ConditionalAffineCoupling, self).__init__()

        self.s = scale
        self.t = shift

        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x, y):
        """Performs a "forward pass" of the coupling block.

        For more information on the affine coupling transform, see the paper
        "Density estimation using Real NVP" by Dinh et al. 2017
        (arXiv:1605.08803).

        Args:
            x: The flow input.
            y: The conditioning input.
        """

        masked = self.mask * x

        s = self.s(masked, y)
        t = self.t(masked, y)
        z = masked + (1 - self.mask) * (x * s.exp() + t)

        log_det = (s * (1 - self.mask)).sum(1)

        return z, log_det

    def reverse(self, z, y):
        """Performs a "reverse pass" of the coupling block.

        For more information on the affine coupling transform, see the paper
        "Density estimation using Real NVP" by Dinh et al. 2017
        (arXiv:1605.08803).

        Args:
            x: The flow input.
            y: The conditioning input.
        """

        masked = self.mask * z

        s = self.s(masked, y)
        t = self.t(masked, y)
        x = masked + (1 - self.mask) * ((z - t) * (-s).exp())

        return x, (-s * (1 - self.mask)).sum(1)