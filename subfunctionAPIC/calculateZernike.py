import warnings
import numpy as np
from subfunctionAPIC.zernike.zernfun2 import zernfun2

def calculateZernike(xsize,ysize,CTF, **kwargs):

    zernikeMode2Opt = np.arange(3, 36)
    CTFThreshold = 1 * 10 ** (-2)

    # Check for valid zernike modes
    for key, value in kwargs.items():
        if key in {'zernikemode', 'zernike mode'}:
            zernikeMode2Opt = value
        elif key in {'max ctf', 'cutoff', 'max freq'}:
            maxCTFUser = value

    if any(zernikeMode2Opt < 3):
        warnings.warn('Warning: The first 3 zernike mode is dropped, as those produce a worse estimation.')
        zernikeMode2Opt = zernikeMode2Opt[zernikeMode2Opt >= 3]

    # Calculate the center coordinates
    xc = xsize // 2 + 1
    yc = ysize // 2 + 1
    nZernike = len(zernikeMode2Opt)
    maxCTF = np.argmax(np.abs(CTF[xc - 1, yc:]) < CTFThreshold) + 1
    # Check for existence of maxCTFUser and compare with maxCTF
    if 'maxCTFUser' in globals():
        if maxCTF < maxCTFUser:
            warnings.warn('Warning: The thresholding-based cutoff frequency estimate is smaller than the given value. '
                          'The code will overwrite this value. Please check the given cutoff freq.')
            maxCTFUser = maxCTF
        temp = np.linspace(-maxCTF / maxCTFUser, maxCTF / maxCTFUser, 2 * maxCTF + 1)
    else:
        temp = np.linspace(-1, 1, 2 * maxCTF + 1)

    Yz, Xz = np.meshgrid(temp, temp)
    theta, r = np.arctan2(Yz, Xz), np.sqrt(Xz ** 2 + Yz ** 2)
    idx2use = r <= 1
    zernikeTemp = zernfun2(zernikeMode2Opt, r[idx2use], theta[idx2use])

    Hz = np.zeros(((2 * maxCTF + 1) ** 2, nZernike))

    # Populate the matrix with Zernike polynomials
    for idx in range(nZernike):
        Hz[idx2use.flatten(), idx] = zernikeTemp[:, idx]
    del zernikeTemp, theta, r

    return Hz
