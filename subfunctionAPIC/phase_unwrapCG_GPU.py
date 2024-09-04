import numpy as np
from scipy.fftpack import dct, idct
import torch
import torch_dct

def phase_unwrapCG_GPU(wrappedPhase, weights=None):
    """
    Performs 2D phase unwrapping using conjugate gradient. (GPU version)

    This implementation is based on the work by Ghiglia and Romero (JOSA A, 1994)
    DOI: 10.1364/JOSAA.11.000107

    Args:
        wrappedPhase (ndarray): The phase to be unwrapped.
        weights (ndarray): Weights to be used in the unwrapping. Ideally, small weights
                           are assigned to places where the phase measurements might be corrupted.

    Returns:
        ndarray: The unwrapped phase.
    """

    # Default weights to ones if not provided
    if weights is None:
        weights = torch.ones_like(wrappedPhase)

    # Normalize the weights
    weights = weights / torch.max(weights)
    sumW = torch.sum(weights)
    if torch.any(weights < 0):
        raise ValueError('Weights cannot be negative. Please check your input.')

    # Preparing for calculating the phase
    xsize, ysize = wrappedPhase.shape
    X, Y = torch.meshgrid(torch.arange(1, xsize + 1, device='cuda'),
                          torch.arange(1, ysize + 1, device='cuda'),indexing = 'ij')
    denominator2use = 2 * (torch.cos(torch.pi * (X - 1) / xsize) + torch.cos(torch.pi * (Y - 1) / ysize) - 2)
    denominator2use[0,0] = 1 # Meaningless value

    # Prepare for the modified discrete Laplacian operator
    squaredW = weights ** 2

    weightsLapX = torch.minimum(squaredW[1:], squaredW[:-1])
    weightsLapX = torch.cat([weightsLapX, torch.zeros(1, squaredW.shape[1], device=weightsLapX.device)], dim=0)
    weightsLapY = torch.minimum(squaredW[:, 1:], squaredW[:, :-1])
    weightsLapY = torch.cat([weightsLapY, torch.zeros(squaredW.shape[0], 1, device=weightsLapY.device)], dim=1)

    dx, dy = dGrad(wrappedPhase)
    dx = wrapToPi(dx)
    dy = wrapToPi(dy)

    unwrappedPhase = torch.zeros_like(wrappedPhase)
    r = dLap(dx, dy, weightsLapX, weightsLapY)
    n0 = torch.norm(r)
    threshold = 1e-8
    # Clear the variable 'p' if it exists
    try:
        del p
    except NameError:
        pass  # p was not defined

    while torch.any(r != 0):
        dctR = dct2(r)
        temp = dctR / denominator2use
        temp[0, 0] = dctR[0, 0]  # Use the same bias
        z = idct2(temp)

        if 'p' not in locals():
            p = z
            rzNow = torch.sum(r * z)
        else:
            rzNow = torch.sum(r * z)
            beta = rzNow / rzPrev
            p = z + beta * p
        rzPrev = rzNow

        dx, dy = dGrad(p)
        dx = wrapToPi(dx)
        dy = wrapToPi(dy)
        Qpk = dLap(dx, dy, weightsLapX, weightsLapY)
        alpha = rzNow / torch.sum(p * Qpk)
        unwrappedPhase = unwrappedPhase + alpha * p
        r = r - alpha * Qpk

        if torch.norm(r) < threshold * n0 or sumW == wrappedPhase.numel():
            break


    return unwrappedPhase


def dGrad(X):
    """
    Calculate the discrete gradient of a matrix.

    Args:
        X (ndarray): A 2D numpy array.

    Returns:
        tuple of ndarray: The discrete gradients along the x-axis (dx) and y-axis (dy).
    """
    dx = X[1:, :] - X[:-1, :]
    dx = torch.cat([dx, torch.zeros(1, X.size(1), device=dx.device)], dim=0)

    dy = X[:, 1:] - X[:, :-1]
    dy = torch.cat([dy, torch.zeros(X.size(0), 1, device=dy.device)], dim=1)

    return dx, dy


def wrapToPi(angles):
    """
    Wrap angle differences to the range [-pi, pi].

    Args:
        angles (ndarray): Array of angles.

    Returns:
        ndarray: Array of wrapped angles.
    """
    return ((angles + torch .pi) % (2 * torch.pi)) - torch.pi


def dLap(dx, dy, wx=None, wy=None):
    """
    Calculate the discrete Laplacian of a matrix.

    Args:
        dx (ndarray): Gradient along the x-axis.
        dy (ndarray): Gradient along the y-axis.
        wx (ndarray, optional): Weights for x-gradient. Defaults to None.
        wy (ndarray, optional): Weights for y-gradient. Defaults to None.

    Returns:
        ndarray: The discrete Laplacian of the matrix.
    """
    L = torch.zeros_like(dx)

    if wx is not None and wy is not None:
        dx = wx * dx
        dy = wy * dy

    Lx = dx[1:, :] - dx[:-1, :]
    Ly = dy[:, 1:] - dy[:, :-1]

    L[1:, :] = Lx
    L[:, 1:] += Ly

    # Handling the boundary conditions
    L[0, :] += dx[0, :]
    L[:, 0] += dy[:, 0]

    return L


# Function for 2D Discrete Cosine Transform
def dct2(a):
    return torch_dct.dct(torch_dct.dct(a.T, norm='ortho').T, norm='ortho')


# Function for 2D Inverse Discrete Cosine Transform
def idct2(a):
    return torch_dct.idct(torch_dct.idct(a.T, norm='ortho').T, norm='ortho')
