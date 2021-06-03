import torch
from abc import ABC, abstractmethod

class Impulse(ABC):
    """A base class for all STEAD impulses.

    A simple base class that defines a required get_impulse method, which is
    used by STEADDataset to produce an impulse waveform.
    """
    @abstractmethod
    def get_impulse(self, start, end, size):
        """A default get_impulse which returns zeros.

        Args:
            start: The start index of the impulse (ignored).
            end: The end index of the impulse (ignored).
            size: The total length of the impulse.
        Returns:
            A torch.Tensor of zeros with shape (1, size).
        """
        return torch.zeros(1, size)


class SpikeImpulse(Impulse):
    """An impulse which is a length one peak.
    
    This class provides an Impulse implementation that produces a "spike" of
    length one at start with given magnitude.

    Attributes:
        magnitude: The magnitude of the impulse spike.
    """
    def __init__(self, magnitude):
        """Initializes a SpikeImpulse with a given magnitude.
        
        Args:
            magnitude: The magnitude of the impulse spike.
        """
        self.magnitude = magnitude

    def get_impulse(self, start, end, size):
        """An impulse that is a length-one spike.

        Args:
            start: The start index of the impulse.
            end: The end index of the impulse (ignored).
            size: The total length of the impulse.
        Returns:
            A torch.Tensor of zeros with shape (1, size), with a single entry
            located at start set to self.magnitude.
        """
        impulse = torch.zeros(1, size)
        if start >= 0:
            impulse[0, start] = self.magnitude
        return impulse


class SquareImpulse(Impulse):
    """An impulse which is a square wave.
    
    This class provides an Impulse implementation that produces a square wave
    of given width and magnitude centered on start and end.

    Attributes:
        width: The half-width (i.e on both sides) of the square wave.
        magnitude: The magnitude of the impulse spike.
    """
    def __init__(self, width, magnitude):
        """Initializes a SquareImpulse with a given width and magnitude.
        
        Args:
            width: The half-width (i.e on both sides) of the square wave.
            magnitude: The magnitude of the square wave.
        """
        self.width = width
        self.magnitude = magnitude

    def get_impulse(self, start, end, size):
        """An impulse that is a square wave.

        Args:
            start: The start index of the impulse.
            end: The end index of the impulse.
            size: The total length of the impulse.
        Returns:
            A torch.Tensor of zeros with shape (1, size). The values between
            start and end, and within self.width on either side. are set to
            self.magnitude, producing a square wave.
        """
        impulse = torch.zeros(1, size)
        if start >= 0:
            lo = max(start - self.width, 0)
            hi = min(end + self.width, size - 1)
            impulse[0, lo:hi] = self.magnitude
        return impulse


class TriangleImpulse(Impulse):
    """An impulse which is a triangle wave.
    
    This class provides an Impulse implementation that produces a triangle
    wave of given width and maximum magnitude centered on start and end.

    Attributes:
        width: The half-width (i.e on both sides) of the triangle wave.
        magnitude: The maximum magnitude of the triangle wave.
    """
    def __init__(self, width, magnitude):
        """Initializes a TriangleImpulse with a given width and magnitude.
        
        Args:
            width: The half-width (i.e on both sides) of the triangle wave.
            magnitude: The maximum magnitude of the triangle wave.
        """

        self.width = width
        self.magnitude = magnitude

    def get_impulse(self, start, end, size):
        """An impulse that is a square wave.

        Args:
            start: The start index of the impulse.
            end: The end index of the impulse.
            size: The total length of the impulse.
        Returns:
            A torch.Tensor of zeros with shape (1, size). The values between
            start and end are set to self.magnitude and the value tapers
            linearly to zero over self.width samples on either side.
        """

        impulse = torch.zeros(1, size)
        if start >= 0:
            impulse[0, start:end] = self.magnitude
            for i in range(self.width):
                lo = max(start - i, 0)
                hi = min(end + i, size - 1)
                mag = self.magnitude * (1.0 - (i / self.width))
                impulse[0, lo] = mag
                impulse[0, hi] = mag
        return impulse


class NoisyImpulse(Impulse):
    """An impulse which adds noise to a provided impulse.
    
    This class provides an Impulse implementation that adds a small amount of
    gaussian noise to every entry of a provided impulse.

    Attributes:
        impulse: The impulse to which noise is added.
        noise: The scale (std. dev.) of gaussian noise to be added.
    """
    def __init__(self, impulse, noise):
        """Initializes a NoisyImpluse with a given impulse and noise.
        
        Args:
            impulse: The impulse to which noise is added.
            noise: The scale (std. dev.) of gaussian noise to be added.
        """

        self.impulse = impulse
        self.noise = noise

    def get_impulse(self, start, end, size):
        """An impulse with noise added.

        Calls self.impulse.get_impulse with the provided start, end, and size,
        then adds gaussian noise as specified by self.noise.

        Args:
            start: The start index of the impulse.
            end: The end index of the impulse.
            size: The total length of the impulse.
        Returns:
            A torch.Tensor corresponding to the impulse provided by
            self.impulse, but with gaussian noise added.
        """

        impulse = self.impulse.get_impulse(start, end, size)
        impulse += self.noise * torch.randn(1, size)
        return impulse


class DownscaledImpulse(Impulse):
    """An impulse that is temporally downscaled.

    This class provides an Impulse implementation that temporally scales the
    provided impulse.

    Attributes:
        impulse: The impulse which is downscaled.
        scale: The downscaling factor to be applied.
    """
    def __init__(self, impulse, scale):
        """Initializes a DownscaledImpulse with a given impulse and scale.
        
        Args:
            impulse: The impulse which is downscaled.
            scale: The downscaling factor to be applied.
        """

        self.impulse = impulse
        self.scale = scale

    def get_impulse(self, start, end, size):
        """An impulse which has been temporally downscaled.

        Scales the start, end, and size down by the given downscaling factor
        before calling self.impulse.get_impulse.

        Args:
            start: The start index of the impulse.
            end: The end index of the impulse.
            size: The total length of the impulse.
        Returns:
            A torch.Tensor of shape (1, size / self.scale) corresponding to a
            temporally downscaled version of self.impulse.
        """

        s_size = int(size / self.scale)
        s_start = int(start / self.scale) if start > 0 else -1
        s_end = int(end / self.scale)

        return self.impulse.get_impulse(s_start, s_end, s_size)
