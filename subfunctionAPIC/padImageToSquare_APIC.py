import numpy as np
import warnings

def pad_image_to_square_apic(I_low, xsize, ysize, xc, yc, freqXY_calib, na_calib):
    """
    Pad measurements to square if the input images are not.
    """
    warnings.warn('For a rectangular image, we scale the pixel related k-vector. '
                  'Also, to improve the efficiency, consider to use a square patch instead.')
    maxsize = max([xsize, ysize])
    numIm = I_low.shape[2]

    # Calculate the ratioXY based on freqXY_calib and na_calib
    ratioXY = (freqXY_calib[:, [1, 0]] - np.array([xc, yc])) / na_calib

    # Padding the images
    I_lowPad = np.zeros((maxsize, maxsize, numIm))
    for idx in range(numIm):
        pad_amount = ((0, maxsize - xsize), (0, maxsize - ysize))
        I_lowPad[:, :, idx] = np.pad(I_low[:, :, idx], pad_amount, mode='constant', constant_values=np.mean(I_low[:, :, idx]))

    xCropPad = xsize
    yCropPad = ysize

    # Check if the ratio is significantly different from the aspect ratio of the crop area
    if abs(np.mean(ratioXY[:, 0]) / np.mean(ratioXY[:, 1]) - yCropPad / xCropPad) > 0.01:
        raise ValueError('For a rectangle image, the illumination k-vector (in pixels) does not match its illumination NA. '
                         'Please make sure the pixel size is correctly set.')

    return I_lowPad, xsize, ysize, xc, yc, freqXY_calib, xCropPad, yCropPad

