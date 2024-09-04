import numpy as np
import torch

def calUnknownMaskFromKnownMask(knownMask, fullMask=None):
    """
    Calculate the unknown part of the entire image/matrix using the known part.
    The second input can be the indicator of what is measured.

    Parameters:
    knownMask: The known part of the mask
    fullMask: Optional. The full mask indicating the entire area. If not provided, it's assumed to be all ones.

    Returns:
    unknownMask: The mask indicating unknown areas.
    linearArea: The area where the linear assumption holds. Returned only if requested.
    """

    # If fullMask is not provided, create one with all ones.
    if fullMask is None:
        fullMask = torch.ones_like(knownMask)
    else:
        fullMask = fullMask.float()

    # Calculate unknown mask
    unknownMask = fullMask - knownMask
    unknownMask = unknownMask == 1

    # Calculate linearArea if requested
    unknownCorrTemp = torch.fft.fftshift(torch.fft.fft2(unknownMask, dim=[-2, -1]), dim=[-2, -1])
    unknownCorr = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(unknownCorrTemp * torch.conj(unknownCorrTemp), dim=[-2, -1]), dim=[-2, -1]), dim=[-2, -1])

    crossCorrTemp = torch.fft.fftshift(torch.fft.fft2(knownMask, dim=[-2, -1]), dim=[-2, -1])
    crossCorr = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(unknownCorrTemp * torch.conj(crossCorrTemp), dim=[-2, -1]), dim=[-2, -1]), dim=[-2, -1])

    linearArea = (torch.real(crossCorr) > 1e-2).float() - (torch.real(unknownCorr) > 1e-2).float()
    linearArea = linearArea == 1

    return unknownMask, linearArea
