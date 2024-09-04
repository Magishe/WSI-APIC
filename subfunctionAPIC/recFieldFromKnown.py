import numpy as np
import torch
import warnings
from subfunctionAPIC.calBoundary import calBoundary
from subfunctionAPIC.phase_unwrapCG_GPU import phase_unwrapCG_GPU
from subfunctionAPIC.exact_shift import exact_shift
from subfunctionAPIC.calUnknownMaskFromKnownMask import calUnknownMaskFromKnownMask
from subfunctionAPIC.calConvMtx_GPU import calConvMtx_GPU
from scipy.ndimage import convolve



def recFieldFromKnown(imStack, kIllu, ftRecons_input, maskRecons_input, CTF_abe, **kwargs):
    """
    recFieldFromKnown reconstruct field based known FT of the measurement
    Input: 1. imStack: measurements from the experiment
           2. kIllu: illumination vector, which corresponds to the shift (in
                     pixel) in spatial frequency domain.
           3. ftRecons: known FT part
           4. maskRecons: mask with 1 denotes the known FT, 0 the unknown
           Options:
              1. drift: when enabled (set to true), the program takes
                        possible calibration error of the illumination vector
                        into consideration
              2. regularization: specify the weight of the L2 regularizer
              3. unknown ratio: only measurements with unknown spectrum that
                                are larger than (this ratio)*CTF will be used.
              4. high freq threshold (threshold): If the unknown spectrum is
                          smaller than this ratio, the newly measured
                          spectrum will be averaged, and will be added to the
                          final spectrum in a later time. The final spectrum
                          is expanded immediately when the unknown spectrum
                          is larger than this ratio.
              5. conserve energy: treat the amplitude of the measured data at
                                  the ground truth.
              6. timer on: calculate the remaining time to finish.

    The source code is licensed under GPL-3.

    By Ruizhi Cao, Nov 11, 2022, Biophotonics Lab, Caltech
    Modified on Oct 31, 2023
    """

    # Default parameter values (equivalent to MATLAB's default values)
    correctDrift = False  # whether to take angle calibration error in to consideration
    pixelTol = 2  # expand the calculated unknown mask by (roughly) pixelTol pixels when drift is enabled
    highFreqTHLD = 0.3  # can be used for an improved reconstruction quality when working with dataset with large overlap ratio.
    valueOnHold = False  # works with nonzero highFreqTHLD
    unknownRatio = 0  # control which measurement to use in the reconstruction.
    # If the ratio unknown part to the (reconstructed) known
    # part is below this threshold, the corresponding measurement
    # is not used in the reconstruction.
    userDefinedReg = False
    userBrightness = False
    autoAmpMatch = False  # whether to correct illumination intensity differences
    marginPixel = 5
    replaceMag = True  # whether to match the energy (intensity) of the complex field reconstruction with the actual measurement.
    ftRecons = ftRecons_input.clone()
    maskRecons = maskRecons_input.clone()
    # Parsing kwargs for optional parameters
    for key, value in kwargs.items():
        key = key.lower().replace("_", "")
        if key in ['drift', 'driftcorrection']:
            correctDrift = value
            if 'pixeltol' in kwargs:
                pixelTol = kwargs['pixeltol']
        elif key in ['reg', 'regularization', 'regularizer']:
            myreg = value
            userDefinedReg = True
        elif key in ['unknownratio', 'minratio']:
            unknownRatio = value
        elif key in ['highfreqthreshold', 'threshold', 'thres']:
            highFreqTHLD = value
            if highFreqTHLD > 0.5:
                warnings.warn('The threshold should not exceed 0.5. Set to 0.5 instead.')
                highFreqTHLD = 0.5
        elif key in ['brightness', 'intensity']:
            myBrightness = value
            userBrightness = True
        elif key in ['intensitycorrection', 'intensitymatch']:
            autoAmpMatch = value
        elif key in ['conserveenergy', 'usedataintensity']:
            replaceMag = value
        else:
            raise ValueError(f"Unsupported argument: {key}")

    if not userDefinedReg:
        # when there is no regularization factor specified, the L2 regularizer's
        # weight is chosen based on whether drift correction is enabled
        if correctDrift:
            myreg = 1  # factor for L2 norm regularizer
        else:
            myreg = 0.01

        # Normalizing CTF if necessary
    maxAmpCTF = torch.max(torch.abs(CTF_abe))
    if maxAmpCTF < 0.99:
        warnings.warn('CTF is unnormalized, it will be normalized by default of the program.')
        CTF_abe = CTF_abe / maxAmpCTF
    CTF = torch.abs(CTF_abe) > 5e-3

    # Checking dimensions of the input arrays
    xsize, ysize, numIm = imStack.shape
    if numIm != len(kIllu[:, 0]):
        raise ValueError('Number of images and number of illumination angles mismatch.')

    if userBrightness and len(myBrightness) != numIm:
        raise ValueError('Number of intensity calibration points disagrees with the number of images.')

    # Creating meshgrid for X and Y coordinates
    X, Y = torch.meshgrid(torch.arange(1, ysize + 1, device='cuda'),
                          torch.arange(1, xsize + 1, device='cuda'),indexing = 'ij')
    xc = np.floor(xsize / 2 + 1).astype('int')
    yc = np.floor(ysize / 2 + 1).astype('int')
    R = torch.abs(X - xc + 1j * (Y - yc))

    # Handling marginPixel and boundary calculation
    if marginPixel > 0.1 * min(xsize, ysize):
        marginPixel = np.ceil(max(0.08 * min(xsize, ysize), 2))
    bdCrop = calBoundary([xc, yc], [xsize - 2 * marginPixel, ysize - 2 * marginPixel])

    # Checking the size of the reconstruction
    tempX, tempY = ftRecons.shape
    if tempX != tempY:
        raise ValueError('The known reconstruction must be a square image.')
    imsizeRecons = tempX
    xcR = np.floor(imsizeRecons / 2 + 1)
    ycR = np.floor(imsizeRecons / 2 + 1)

    # Handling maxCTF
    if 'maxCTF' not in locals():
        maxCTF = torch.round((torch.sum(CTF[int(xc)-1, :]) - 1) / 2+1e-3)
        if torch.sum(CTF[int(xc), :]) != torch.sum(CTF[:, int(yc)]):
            raise ValueError('The input image is not a square image. Please use zero padding.')

    if correctDrift:
        CTF_larger = (R < maxCTF + 1)

    areaCTF = torch.sum(CTF)
    # Initialization for high frequency threshold
    if highFreqTHLD != 0:
        unknownTHLD = highFreqTHLD * areaCTF  # threshold in terms of the area of the unknown mask
        ftExpanded = torch.zeros(imsizeRecons, imsizeRecons, dtype=torch.complex64).cuda()
        maskExpanded = torch.zeros(imsizeRecons, imsizeRecons, dtype=torch.float32).cuda()# number of repeats for the expanded spectrum

    # Calculating boundary
    bd = calBoundary([xcR, ycR], [xsize, ysize])
    for idx in range(numIm):
        flagSubPixel = False
        bd2use = bd.astype(np.float32) - np.tile(kIllu[idx, :], (2, 1))
        if np.any(np.mod(bd2use, 1) != 0):
            temp = bd2use - np.round(bd2use)
            subpixelShift = -np.array([temp[0,0], temp[0,1]])
            bd2use = np.round(bd2use).astype(int)
            flagSubPixel = True

        # Introduce aberration to the known part to match up with real measurement
        if flagSubPixel:
            CTF_abe_copy = CTF_abe # In case that CTF and subpixelShift may be changed
            subpixelShift_copy = subpixelShift.copy()
            ampCTF = exact_shift(torch.abs(CTF_abe_copy), -subpixelShift_copy, isRealSpace=True)
            CTF_abe_copy = CTF_abe  # In case that CTF and subpixelShift may be changed
            subpixelShift_copy = subpixelShift.copy()
            A = phase_unwrapCG_GPU(torch.angle(CTF_abe_copy))
            aglCTF = exact_shift(A, -subpixelShift_copy, isRealSpace=True)
            ampCTF = ampCTF * (ampCTF > 0.12)
            ampCTF = ampCTF * (ampCTF <= 1) + (ampCTF > 1)
            CTF2use = ampCTF * torch.exp(1j * aglCTF)
        else:
            if idx == 0:
                CTF2use = CTF_abe * (torch.abs(CTF_abe) > 1e-3)
                ampCTF = torch.abs(CTF2use)


        lowFT = ftRecons[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] * CTF2use
        knownMask = maskRecons[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] * CTF
        unknownMask, linearArea = calUnknownMaskFromKnownMask(knownMask, CTF)

        if highFreqTHLD != 0 and torch.sum(unknownMask) > unknownTHLD:
            # Expand the spectrum of the reconstruction
            ftRecons += ftExpanded / (maskExpanded + torch.finfo(torch.float32).eps)
            maskRecons += (maskExpanded != 0)

            # Reset temporary spectrum and its weight mask
            maskExpanded.fill_(0)
            ftExpanded.fill_(0)
            valueOnHold = False

            # Regenerate the known field and the known and unknown masks
            lowFT = ftRecons[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] * CTF2use
            knownMask = maskRecons[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] * CTF
            unknownMask, linearArea = calUnknownMaskFromKnownMask(knownMask, CTF)

        if torch.sum(unknownMask) > unknownRatio * areaCTF:
            fieldKnown = torch.fft.ifft2(torch.fft.ifftshift(lowFT, dim=[-2, -1]), dim=[-2, -1])
            imKnown = fieldKnown * torch.conj(fieldKnown)

            # Calculate the correlation to correct the intensity
            if autoAmpMatch:
                imKnownCrop = imKnown[bdCrop[0,0]-1:bdCrop[1,0], bdCrop[0,1]-1:bdCrop[1,1]]
                imStackCrop = imStack[bdCrop[0,0]-1:bdCrop[1,0], bdCrop[0,1]-1:bdCrop[1,1], idx]
                meanImKnownCrop = torch.mean(imKnownCrop)
                meanImStackCrop = torch.mean(imStackCrop)
                corrF = torch.sum((imKnownCrop - meanImKnownCrop) * (imStackCrop - meanImStackCrop)) / \
                        torch.sum((imStackCrop - meanImStackCrop) ** 2)
            else:
                corrF = 1

            if userBrightness:
                corrF *= myBrightness[idx]

            imReal = imStack[:, :, idx] * corrF - imKnown
            ftImSub = torch.fft.fftshift(torch.fft.fft2(imReal, dim=[-2, -1]), dim=[-2, -1]) * xsize * ysize
            confinedCTFBoxSize = np.int_((maxCTF.cpu() + 1))  # 1 more point as residual


            # Vertices of a smaller square that contains the CTF
            boxVert = [(xc - confinedCTFBoxSize), (xc + confinedCTFBoxSize),
                       (yc - confinedCTFBoxSize), (yc + confinedCTFBoxSize)]

            # Check for aliasing
            if any(v < 1 for v in boxVert) or boxVert[1] > xsize or boxVert[3] > ysize:
                raise ValueError('There is aliasing in the captured image, please reduce the CTF size.')

            if correctDrift:
                kernelTol = torch.ones((1, 1, 2 * pixelTol + 1, 2 * pixelTol + 1), dtype=torch.float32).cuda()
                unknownMaskOriginal = unknownMask.clone()
                unknownMask_reshaped = unknownMask.unsqueeze(0).unsqueeze(0)
                temp = torch.nn.functional.conv2d(unknownMask_reshaped.float(), kernelTol, padding=pixelTol)
                temp = (temp > 0.5) * CTF_larger
                temp = temp.squeeze(0).squeeze(0)

                unknownMask[boxVert[0]:boxVert[1], boxVert[2]:boxVert[3]] = \
                    temp[boxVert[0]:boxVert[1], boxVert[2]:boxVert[3]]

            A = torch.rot90(torch.conj(lowFT[boxVert[0]-1:boxVert[1], boxVert[2]-1:boxVert[3]]), 2)
            B = linearArea[(xc - confinedCTFBoxSize * 2)-1:(xc + confinedCTFBoxSize * 2),
                                  (yc - confinedCTFBoxSize * 2)-1:(yc + confinedCTFBoxSize * 2)]
            C = unknownMask[boxVert[0]-1:boxVert[1], boxVert[2]-1:boxVert[3]]
            Hreduced = calConvMtx_GPU(A,B,C)

            Htemp = torch.mm(torch.conj(Hreduced.T), Hreduced)
            absMean = torch.mean(torch.abs(Htemp))


            if correctDrift:
                vecWeight = (torch.diag(unknownMaskOriginal.T[unknownMask.T].float()) + 0.00001) / 1.00001
                recFTvct = torch.linalg.solve(Htemp + absMean * myreg * vecWeight,
                                              torch.mv(torch.conj(Hreduced.T), ftImSub.T[(linearArea == 1).T]))
            else:
                recFTvct = torch.linalg.solve(Htemp + absMean * myreg * torch.eye(Htemp.shape[0]).cuda(),
                                              torch.mv(torch.conj(Hreduced.T), ftImSub.T[(linearArea == 1).T]))
            del Htemp, Hreduced
            torch.cuda.empty_cache()
            # Reconstruct ftTrue
            ftTrue = torch.zeros(xsize, ysize, dtype=torch.complex64).cuda()
            ftTrue.T[unknownMask.T] = recFTvct

            # Correct for drift if necessary
            if correctDrift:
                unknownMask = unknownMaskOriginal
                ftTrue *= unknownMaskOriginal

            # Replace magnitude if necessary
            if replaceMag:  # whether to maintain the (pixel-wise) energy
                fieldTemp = torch.exp(1j * torch.angle(torch.fft.ifft2(torch.fft.ifftshift(ftTrue + lowFT, dim=[-2, -1]),dim=[-2, -1]))) * torch.sqrt(
                    imStack[:, :, idx] * corrF)
                ftTemp = torch.fft.fftshift(torch.fft.fft2(fieldTemp, dim=[-2, -1]), dim=[-2, -1])
                ftTrue[unknownMask] = ftTemp[unknownMask]

            # Correct aberration
            ftTrue *= torch.conj(CTF2use) / (torch.abs(CTF2use) + 0.005) * (ampCTF > 0.05) * 1.005

            if highFreqTHLD == 0:
                # Update ftRecons and maskRecons if highFreqTHLD is zero
                ftRecons[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] += ftTrue
                maskRecons[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] += unknownMask * (ampCTF > 0.05)
            else:
                # Update ftExpanded and maskExpanded otherwise
                ftExpanded[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] += ftTrue
                maskExpanded[bd2use[0,0]-1:bd2use[1,0], bd2use[0,1]-1:bd2use[1,1]] += unknownMask * (ampCTF > 0.05)
                valueOnHold = True
            MeanValue = torch.mean(torch.abs(ftExpanded))




    if valueOnHold:
        # Update ftRecons by adding values from ftExpanded, with division by maskExpanded + a small epsilon
        ftRecons += ftExpanded / (maskExpanded + torch.finfo(torch.float32).eps)

        # Update maskRecons by adding 1 where maskExpanded is not zero
        maskRecons += (maskExpanded != 0)

    return ftRecons, maskRecons
