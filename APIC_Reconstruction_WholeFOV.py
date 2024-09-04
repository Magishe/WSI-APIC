import os
import glob
import time
import argparse
import h5py

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import zoom
import scipy.io as sio
import torch

from subfunctionAPIC.preprocessing import preprocessing
from subfunctionAPIC.padImageToSquare_APIC import pad_image_to_square_apic
from subfunctionAPIC.calBoundary import calBoundary
from subfunctionAPIC.recFieldKK import recFieldKK
from subfunctionAPIC.findAbeFromOverlap import findAbeFromOverlap
from subfunctionAPIC.cNeoAlbedo import cNeoAlbedo
from subfunctionAPIC.cDivVlag import cDivVlag
from subfunctionAPIC.recFieldFromKnown import recFieldFromKnown


# from scipy.ndimage import zoom


def main():
    # Options for the reconstruction
    parser = argparse.ArgumentParser()
    parser.add_argument("--saveResult",default=True,type=bool, help="whether to save the reconstruction results")
    parser.add_argument("--matchingTol",default=0.03,type=float, help="maximal tolerance in the deviation between the illumination NA of the NA-matching measurement and the objective NA")
    parser.add_argument("--useAbeCorrection", default=True, type=bool, help="whether to use aberration correction")
    parser.add_argument("--enableROI", default=True, type=bool,
                        help="whether to use ROI in the reconstruction, this is the reconstruction size used in the code")
    parser.add_argument("--patchNumber", default=5, type=int, help="the total number of patches into which you intend to divide the full field of view (FOV) along one dimension")
    parser.add_argument("--overlappingSize", default=20, type=int, help="the overlap size between different patches (for stitching inside one FOV)")
    parser.add_argument("--paddingHighRes", default=3, type=int, help="define the upsampling ratio for the final high-res image")
    parser.add_argument("--visualizePupil", default=True, type=bool, help="whether to visualize the reconstructed pupil")
    # Options for data loading
    parser.add_argument("--folderName", default='Data', type=str, help="Folder name, note this is case sensitive")
    parser.add_argument("--fileNameKeyword", default='NSCLC_Scan', type=str, help="e.g.: Siemens HEpath Thyroid")
    parser.add_argument("--additionalKeyword", default='_20_19_g', type=str, help=" # this is only needed when there are multiple files containing the keyword in the above line")

    # Options for pre-processing
    parser.add_argument("--preprocessed", default=False, type=bool, help="whether the data is preprocessed")
    parser.add_argument("--preprocessed_file",default="Data\Calibration\JoshDF_New_System_Modifed_CalibrateDF_Reduced.mat", type=str,help="Calibration file")
    parser.add_argument("--flag_HPR", default=True, type=bool, help="whether to remove hot pixel")
    parser.add_argument("--flag_show", default=False, type=bool, help="whether to show the calibration results")
    parser.add_argument("--patch_size", default=2560, type=int, help="patch_size for preprocess")
    parser.add_argument("--na_cal", default=0.26, type=float, help="Objective NA for NA matching")
    parser.add_argument("--dark_th", default=300, type=float, help="dark_th to be regarded as hot pixel")

    # Options for recFieldKK
    parser.add_argument("--KK_wiener", default=False, type=bool, help="whether to use Wiener filter to mitigate noise")
    parser.add_argument("--KK_norm", default=True, type=bool, help="whether to normalize the acquired images such that they have the same effective intensity (mean value)")
    parser.add_argument("--KK_pad", default=3, type=int, help="Choose the zero-padding factor of the FT of the images")

    # Options for findAberration
    parser.add_argument("--Abe_weighted", default=True, type=bool, help="whether to use weighted matrix in the algorithm, in which case the algorithm focuses more on larger signals")

    # Options for recFieldFromKnown
    parser.add_argument("--DF_drift", default=True, type=bool, help="whether to take possible calibration error of the illumination vector into consideration")
    parser.add_argument("--DF_threshold", default=0.3, type=float, help="The final spectrum is expanded immediately when the unknown spectrum is larger than this ratio")
    parser.add_argument("--DF_reg", default=0.01, type=float, help="specify the weight of the L2 regularizer")
    parser.add_argument("--DF_intensity_correction", default=True, type=bool, help="whether to calculate the correlation to correct the intensity")
    parser.add_argument("--DF_use_data_intensity", default=True, type=bool, help="whether to use data intensity")

    args = parser.parse_args()
    saveResult = args.saveResult
    matchingTol = args.matchingTol
    enableROI = args.enableROI
    useAbeCorrection = args.useAbeCorrection
    overlapping_size = args.overlappingSize
    patchNumber = args.patchNumber
    paddingHighRes = args.paddingHighRes
    visualizePupil = args.visualizePupil
    folderName = args.folderName
    fileNameKeyword = args.fileNameKeyword
    additionalKeyword = args.additionalKeyword
    preprocessed = args.preprocessed
    preprocessed_file = args.preprocessed_file
    flag_HPR = args.flag_HPR
    flag_show = args.flag_show
    patch_size = args.patch_size
    na_cal = args.na_cal
    dark_th = args.dark_th
    KK_wiener = args.KK_wiener
    KK_norm = args.KK_norm
    KK_pad = args.KK_pad
    Abe_weighted = args.Abe_weighted
    DF_drift = args.DF_drift
    DF_threshold = args.DF_threshold
    DF_reg = args.DF_reg
    DF_intensity_correction = args.DF_intensity_correction
    DF_use_data_intensity = args.DF_use_data_intensity


    if not os.path.exists(folderName):  # Check if the folder exists
        raise FileNotFoundError(f"No folder with name '{folderName}' under current directory.")
    filePattern = os.path.join(folderName, f"*{fileNameKeyword}*.mat")
    fileList = glob.glob(filePattern)

    if not fileList:  # Error handling for file existence and uniqueness
        raise FileNotFoundError(f"No .mat file with name '{fileNameKeyword}' found in folder {folderName}")
    elif len(fileList) > 1:
        fileList = glob.glob(os.path.join(folderName, f"*{fileNameKeyword}*{additionalKeyword}*.mat"))
        if len(fileList) > 1:
            raise FileExistsError("Multiple raw data files found in the folder. Consider specifying the full name or "
                                  "check the additional keyword.")

    fileName = fileList[0]  # Load the data from the first file in the list
    if os.path.exists(fileName):
        if not preprocessed:
            with h5py.File(preprocessed_file, 'r') as data_cal:
                LED_seq = data_cal["LED_seq"][:]
                dpix_c = data_cal["dpix_c"][:][0, 0]
                num_matched = round(data_cal["num_matched"][:][0, 0])
                ring_radius = data_cal["ring_radius"][:][0, 0]
                mag = data_cal["mag"][:][0, 0]
                NA = data_cal["NA"][:][0, 0]
                NA_seq = data_cal["NA_seq"][:]
                ringIdx = data_cal["ringIdx"][:][0, 0]
                height = data_cal["height"][:]
            LED_seq = LED_seq.transpose(1, 0)
            NA_seq = NA_seq.transpose(1, 0)

            with h5py.File(fileName, 'r') as data:
                darkframe = data["darkframe"][:]
                exposure_DF = data["exposure_DF"][:][0, 0]
                exposure_Ring = data["exposure_Ring"][:][0, 0]
                imlow_BF_IN = data["imlow_BF_IN"][:]
                lambda_g = data["lambda"][:][0, 0]

            imlow_BF_IN = imlow_BF_IN.transpose(2, 1, 0)
            darkframe = darkframe.transpose(1, 0)
            # Performing preprocess for dark noise and hot pixel removal. preprocessed_file should be changed to your own calibration files.
            I_low, na_calib, na_cal, freqXY_calib, na_rp_cal = preprocessing(NA_seq, imlow_BF_IN, na_cal, patch_size,
                                                                             dark_th, darkframe, num_matched, exposure_DF,
                                                                             exposure_Ring,
                                                                             lambda_g, dpix_c, mag, NA, preprocessed,
                                                                             flag_HPR, flag_show)
        else:
            with h5py.File(fileName, 'r') as data:
                I_low = data["I_low"][:]
                dpix_c = data["dpix_c"][:]
                freqXY_calib = data["freqXY_calib"][:]
                wavelength = data["lambda"][:]
                mag = data["mag"][:]
                na_cal = data["na_cal"][:]
                na_calib = data["na_calib"][:]
                na_rp_cal = data["na_rp_cal"][:]
            I_low = I_low.transpose(2, 1, 0)
            freqXY_calib = freqXY_calib.transpose(1, 0)
            na_calib = na_calib.transpose(1, 0)


        batch_number = patchNumber
        imsize = I_low.shape[1]
        batch_size = round(imsize/batch_number)
        ROILength = batch_size + overlapping_size
        himAPIC_FullField = I_low.shape[1]*paddingHighRes
        himAPIC_stack = torch.zeros([himAPIC_FullField,himAPIC_FullField],dtype=torch.complex64).cuda()
        for cropi in range(batch_number):
            for cropj in range(batch_number):
                ROIcenter = [batch_size/2+1+cropi*batch_size,batch_size/2+1+cropj*batch_size]
                ## Select measurement whose illumination NA matches up with the objective NA
                NA_diff = np.abs(np.sqrt(na_calib[:, 0] ** 2 + na_calib[:, 1] ** 2) - na_cal)  # Calculate the absolute difference between illumination NA and objective NA
                NA_diff = NA_diff.T
                slt_idx = np.where(NA_diff < matchingTol)[0]  # Select the NA matching measurements based on the tolerance threshold
                nNAmatching = len(slt_idx)
                xsize, ysize = I_low.shape[0:2]
                xc = np.floor(xsize / 2 + 1)
                yc = np.floor(ysize / 2 + 1)
                counting_stack = torch.ones([ROILength*paddingHighRes, ROILength*paddingHighRes], device='cuda')
                overlap = overlapping_size*paddingHighRes
                for j in range(overlap):
                    alpha = j / overlap
                    counting_stack[:, j] = alpha
                    counting_stack[:, -j - 1] = alpha
                for i in range(overlap):
                    alpha = i / overlap
                    counting_stack[i, :] = torch.min(counting_stack[i, :],
                                                     torch.full([ROILength*paddingHighRes], alpha, device='cuda'))
                    counting_stack[-i - 1, :] = torch.min(counting_stack[-i - 1, :],
                                                          torch.full([ROILength*paddingHighRes], alpha, device='cuda'))

                if ROILength > xsize or ROILength > ysize:
                    raise ValueError(f"ROI length cannot exceed {min(xsize, ysize)}")
                if not enableROI and xsize != ysize:
                    # Assuming pad_image_to_square_apic function is already defined as translated earlier
                    I_low, xsize, ysize, xc, yc, freqXY_calib, xCropPad, yCropPad = pad_image_to_square_apic(I_low, xsize, ysize, xc, yc, freqXY_calib, na_calib)
                # Get the calibrated illumination angles for NA-matching measurements
                x_illumination = freqXY_calib[slt_idx, 1]  # Note the second column is assi
                # gned to x
                y_illumination = freqXY_calib[slt_idx, 0]  # Note the first column is assigned to y
                NA_pixel = na_rp_cal  # Calibrated maximum spatial freq in FT space
                print(f"Number of NA-matching measurements found: {nNAmatching}")

                ## Select dark field measurement
                NA_diff = np.abs(na_calib[:, 0] + 1j * na_calib[:, 1])  - na_cal
                NA_diff = NA_diff.T
                # Find the indices where NA_diff is greater than 0
                slt_idxDF = np.where(NA_diff > 0)[0]

                # LED illumination angle, darkfield measurements
                x_illumination = np.append(x_illumination, freqXY_calib[slt_idxDF, 1])
                y_illumination = np.append(y_illumination, freqXY_calib[slt_idxDF, 0])

                # Change center to where the zero frequency is
                x_illumination -= xc
                y_illumination -= yc

                # Scaling x_illumination, y_illumination, and NA_pixel based on ROILength if enableROI is True
                if enableROI:
                    x_illumination *= ROILength / xsize
                    y_illumination *= ROILength / ysize
                    NA_pixel *= ROILength / xsize  # Assuming NA_pixel is defined

                    # Depending on the type of ROIcenter, calculate the boundary using calBoundary function
                    if isinstance(ROIcenter, (list, np.ndarray)):
                        bdROI = calBoundary(ROIcenter, ROILength)
                        for columj in range(bdROI.shape[1]):
                            if bdROI[0, columj] < 1:
                                bdROI[1, columj] = 1 - bdROI[0, columj] + bdROI[1, columj]
                                bdROI[0, columj] = 1

                        for columj in range(bdROI.shape[1]):
                            if bdROI[1, columj] > imsize:
                                bdROI[0, columj] = bdROI[0, columj] - (bdROI[1, columj] - xsize)
                                bdROI[1, columj] = imsize
                    elif ROIcenter.lower() == 'auto':
                        bdROI = calBoundary([xc, yc], ROILength)  # ROI locates in the center of the image
                    else:
                        raise ValueError("ROIcenter should be a 1-by-2 vector or 'auto'.")

                    # Check if the boundary exceeds the image size
                    if (bdROI < 1).any() or bdROI[0, 1] > xsize or bdROI[1, 1] > ysize:
                        raise ValueError("ROI exceeds the boundary. Please check ROI's center and length")

                    # Update xsize and ysize to ROILength
                    xsize = ysize = ROILength
                else:
                    # By default, use the maximum ROI
                    bdROI = np.array([1, 1, xsize, ysize])

                # Selecting a subset of I_low based on bdROI and indices slt_idx, slt_idxDF. Using numpy's advanced indexing
                I_test = imlow_BF_IN[bdROI[0, 0] - 1:bdROI[1, 0], bdROI[0, 1] - 1:bdROI[1, 1], 0]
                var_I = np.var(I_test)

                if var_I > 50000: ## filter out the background region
                    I = I_low[bdROI[0, 0] - 1:bdROI[1, 0], bdROI[0, 1] - 1:bdROI[1, 1], :][:, :, np.r_['-1', slt_idx, slt_idxDF]]

                    ## Preparing for Reconstruction
                    imStack = np.zeros_like(I)  # Allocate space for the filtered measurement

                    # Order measurement under NA-matching angle illumination
                    theta = np.arctan2(y_illumination[:nNAmatching], x_illumination[:nNAmatching])
                    pupilR = np.sqrt(x_illumination[:nNAmatching] ** 2 + y_illumination[:nNAmatching] ** 2)

                    idxMap = np.argsort(theta)

                    # Order dark field measurements
                    pupilR_DF = np.sqrt(x_illumination[nNAmatching:] ** 2 + y_illumination[nNAmatching:] ** 2)
                    idxMapDF = np.argsort(pupilR_DF) + nNAmatching
                    idxAll = np.concatenate((idxMap, idxMapDF))

                    # Calculate Maximum Spatial Frequency
                    enlargeF = 4
                    Y, X = np.meshgrid(range(1, ysize * enlargeF + 1), range(1, xsize * enlargeF + 1))
                    xc = xsize * enlargeF // 2 + 1
                    yc = ysize * enlargeF // 2 + 1
                    R_enlarge = np.abs(X - xc + 1j * (Y - yc))

                    k_illu = np.column_stack((x_illumination[idxAll], y_illumination[idxAll]))

                    Y, X = np.meshgrid(range(1, ysize + 1), range(1, xsize + 1))
                    xc = xsize // 2 + 1
                    yc = ysize // 2 + 1
                    R = np.abs(X - xc + 1j * (Y - yc))

                    pupilRadius = max([NA_pixel, np.max(pupilR), np.linalg.norm(np.fix(np.column_stack((x_illumination[:nNAmatching], y_illumination[:nNAmatching]))), axis=1).max()])

                    CTF_Unresized = (R_enlarge < pupilRadius * enlargeF).astype('float32')
                    im = Image.fromarray(CTF_Unresized)
                    CTF = np.array(im.resize((xsize, ysize), Image.BILINEAR))
                    # CTF = zoom((R_enlarge < pupilRadius * enlargeF).astype('float32'), 1/enlargeF, order=1)
                    CTF = np.maximum(np.roll(np.rot90(CTF, 2), (xsize % 2, ysize % 2), axis=(0, 1)), CTF)
                    binaryMask = R <= 2 * pupilRadius

                    # Taper Edge to Avoid Ringing Effect
                    edgeMask = np.zeros((xsize, ysize))
                    pixelEdge = 3
                    edgeMask[:pixelEdge, :] = edgeMask[-pixelEdge:, :] = edgeMask[:, :pixelEdge] = edgeMask[:, -pixelEdge:] = 1
                    edgeMask = gaussian_filter(edgeMask, 5, mode="nearest", truncate=2.0)  # This is equivalent to matlab imgaussfilt
                    maxEdge = edgeMask.max()
                    edgeMask = (maxEdge - edgeMask) / maxEdge

                    # Noise Level Calculation and Image Stack Generation
                    noiseLevel = np.zeros(len(idxAll))
                    for idx in range(len(idxAll)):
                        ftTemp = fftshift(fft2(I[:, :, idxAll[idx]]))
                        noiseLevel[idx] = max([np.finfo(float).eps, np.mean(np.abs(ftTemp[~binaryMask]))])
                        ftTemp *= np.abs(ftTemp) / (np.abs(ftTemp) + noiseLevel[idx])
                        if idx > nNAmatching - 1:
                            imStack[:, :, idx] = np.real(ifft2(ifftshift(ftTemp * binaryMask)) * edgeMask)
                        else:
                            imStack[:, :, idx] = np.real(ifft2(ifftshift(ftTemp * binaryMask)))

                    del I  # Clear I

                    # Allocate Space for High-Resolution Image
                    imsizeRecons = paddingHighRes * xsize
                    ftRecons = torch.zeros(imsizeRecons, imsizeRecons, dtype=torch.complex64).cuda()
                    maskRecons = torch.zeros(imsizeRecons, imsizeRecons, dtype=torch.float32).cuda()

                    # Center and Grid of the Final High-Res Image
                    xcR = imsizeRecons // 2 + 1
                    ycR = imsizeRecons // 2 + 1
                    YR, XR = np.meshgrid(range(1, imsizeRecons + 1), range(1, imsizeRecons + 1))
                    R_recons = np.abs((XR - xcR) + 1j * (YR - ycR))
                    # Clearing unnecessary variables
                    del YR, XR

                    imStack_cuda = torch.tensor(imStack[:, :, 0:nNAmatching].copy().astype(np.float32)).cuda()
                    k_illu_cpu = k_illu[0:nNAmatching, :].copy().astype(np.float32)
                    CTF_cuda = torch.tensor(CTF.copy().astype(np.float32)).cuda()
                    imStack_cuda_Dark = torch.tensor(imStack[:, :, nNAmatching:].astype(np.float32)).cuda()
                    k_illu_cpu_Dark = k_illu[nNAmatching:, :].astype(np.float32)

                    ## field reconstruction of NA-matching angle measurements and aberration extraction
                    timestart = time.time()
                    recFTframe, mask2use = recFieldKK(imStack_cuda, k_illu_cpu, ctf=CTF_cuda, pad=KK_pad, norm=KK_norm, wiener=KK_wiener)  # recFTframe: reconstructed complex spectrums of NA-matching measurements
                    timeKK = time.time()

                    # Find Aberration CPU
                    # convert to cpu arrays
                    k_illu_cpu = k_illu[0:nNAmatching, :].copy().astype(np.float32)
                    recFTframe_cpu = np.array(recFTframe.cpu()).astype(np.complex128)
                    CTF_abe, zernikeCoeff = findAbeFromOverlap(recFTframe_cpu, k_illu_cpu, CTF, weighted=Abe_weighted)
                    # convert back to gpu arrays
                    CTF_abe_abs_cuda = torch.tensor(np.abs(CTF_abe)).cuda()
                    CTF_abe_cuda = torch.tensor(CTF_abe.astype(np.complex64)).cuda()
                    zernikeCoeff_cuda = torch.tensor(zernikeCoeff.astype(np.float32)).cuda()
                    timeFindAbe = time.time()

                    # correct and stitch the reconstructed spectrums using NA-matching measurements
                    bd = calBoundary([xcR, ycR], [xsize, ysize])
                    normMask = torch.zeros(maskRecons.shape).cuda()
                    maskRecons[xcR - 1, ycR - 1] = 1

                    k_illu_cpu = k_illu.copy().astype(np.float32)
                    X, Y = torch.meshgrid(torch.arange(1, ysize + 1, device='cuda'), torch.arange(1, xsize + 1, device='cuda'),indexing='ij')

                    for idx in range(nNAmatching):
                        bd2use = bd - np.tile(np.round(k_illu_cpu[idx, :]), (2, 1)).astype('int')
                        maskOneside = (-(X - xc - k_illu_cpu[idx, 0]) * k_illu_cpu[idx, 0] -
                                       (Y - yc - k_illu_cpu[idx, 1]) * k_illu_cpu[idx, 1] >
                                       -0.5 * torch.norm(torch.tensor(k_illu_cpu[idx, :])))
                        mask2useNew = mask2use * maskOneside  # mask to filter the reconstructed spectrums

                        unknownMask = (1 - maskRecons[bd2use[0, 0] - 1:bd2use[1, 0], bd2use[0, 1] - 1:bd2use[1, 1]]) * mask2use
                        maskRecons[bd2use[0, 0] - 1:bd2use[1, 0], bd2use[0, 1] - 1:bd2use[1, 1]] += unknownMask
                        A = CTF_abe_cuda[xc + round(k_illu_cpu[idx, 0]) - 1, yc + round(k_illu_cpu[idx, 1]) - 1].cpu().numpy()
                        offsetPhase = np.angle(A)

                        normMask[bd2use[0, 0] - 1:bd2use[1, 0], bd2use[0, 1] - 1:bd2use[1, 1]] += mask2useNew
                        if useAbeCorrection:
                            A = recFTframe[:, :, idx] * torch.conj(CTF_abe_cuda) * np.exp(1j * offsetPhase)
                            B = CTF_abe_abs_cuda + 1e-3
                            C = A/B
                            ftRecons[bd2use[0, 0] - 1:bd2use[1, 0], bd2use[0, 1] - 1:bd2use[1, 1]] += C

                        else:
                            ftRecons[bd2use[0, 0] - 1:bd2use[1, 0], bd2use[0, 1] - 1:bd2use[1, 1]] += (recFTframe[:, :, idx] * mask2use / (mask2use + 1e-3))

                    normMask[xcR - 1, ycR - 1] = nNAmatching
                    ftRecons = ftRecons * (normMask > 0.5) / (normMask + 1e-5) * maskRecons
                    himMatching = torch.fft.ifft2(torch.fft.ifftshift(ftRecons, dim=[-2, -1]),dim=[-2, -1])  # reconstructed complex field using NA-matching measurements


                    edgeMask_tensor = torch.from_numpy(edgeMask).to(torch.float32).cuda()
                    tempMask = torch.nn.functional.interpolate(edgeMask_tensor.unsqueeze(0).unsqueeze(0),
                                                               size=(ftRecons.shape[0], ftRecons.shape[1]),
                                                               mode='bilinear', align_corners=False).squeeze()

                    ftRecons = torch.fft.fftshift(torch.fft.fft2(himMatching, dim=[-2, -1]), dim=[-2, -1])
                    timeFindAbe_Finish = time.time()
                    ftRecons, maskRecons = recFieldFromKnown(imStack_cuda_Dark, k_illu_cpu_Dark, ftRecons, maskRecons,
                                                             CTF_abe_cuda,
                                                             drift=DF_drift, threshold=DF_threshold, reg=DF_reg,
                                                             intensity_correction=DF_intensity_correction,
                                                             use_data_intensity=DF_use_data_intensity)

                    himAPIC_Temp = torch.fft.ifft2(torch.fft.ifftshift(ftRecons, dim=[-2, -1]), dim=[-2, -1])
                    timeDF_End = time.time()
                    print(f'Total reconstruction time for one patch: {timeDF_End - timestart}')

                    if visualizePupil:
                        plt.figure()
                        cNeoAlbedo_colormap = cNeoAlbedo()
                        plt.imshow(np.angle(CTF_abe * (np.abs(CTF_abe) > 1e-3)), cmap=cNeoAlbedo_colormap,
                                   vmin=-np.pi, vmax=np.pi)

                        plt.axis('off')
                        plt.title('Reconstructed pupil, APIC')
                        plt.colorbar()
                        plt.savefig("E:\Fast_Scanner_New\Figures\Figure_6/pupil_b5.png", dpi=300)
                        plt.show()

                else:
                    himAPIC_Temp = torch.zeros([ROILength * paddingHighRes, ROILength * paddingHighRes], device='cuda')

                ## Stitching
                himAPIC_Tempsize = [[1, 1],[himAPIC_Temp.shape[0], himAPIC_Temp.shape[1]]]
                counting_stack_Tempsize = [[1, 1],[himAPIC_Temp.shape[0], himAPIC_Temp.shape[1]]]
                for columj in range(bdROI.shape[1]):  # Iterate over columns
                    if bdROI[0, columj] == 1:
                        bdROI[1, columj] -= overlapping_size / 2
                        counting_stack_Tempsize[0][columj] += round((overlapping_size / 2) * paddingHighRes)
                        himAPIC_Tempsize[1][columj] -= round((overlapping_size / 2) * paddingHighRes)
                        if columj == 0:
                            counting_stack[0:overlap, overlap:ROILength * paddingHighRes - overlap] = 1
                        else:
                            counting_stack[overlap:ROILength * paddingHighRes - overlap, 0:overlap] = 1
                if bdROI[0, 0] == 1 and bdROI[0, 1] == 1:
                    counting_stack[0:overlap, 0:overlap] = 1
                for columj in range(bdROI.shape[1]):  # Iterate over columns
                    if bdROI[1, columj] == imsize:
                        bdROI[0, columj] += overlapping_size / 2
                        counting_stack_Tempsize[1][columj] -= round((overlapping_size / 2) * paddingHighRes)
                        himAPIC_Tempsize[0][columj] += round((overlapping_size / 2) * paddingHighRes)
                        if columj == 0:
                            counting_stack[ROILength * paddingHighRes - overlap:ROILength * paddingHighRes + 1,
                            overlap:ROILength * paddingHighRes - overlap] = 1
                        else:
                            counting_stack[overlap:ROILength * paddingHighRes - overlap,
                            ROILength * paddingHighRes - overlap:ROILength * paddingHighRes + 1] = 1
                if bdROI[1, 0] == imsize and bdROI[1, 1] == imsize:
                    counting_stack[ROILength * paddingHighRes - overlap:ROILength * paddingHighRes + 1,
                    ROILength * paddingHighRes - overlap:ROILength * paddingHighRes + 1] = 1
                if bdROI[1, 0] == imsize and bdROI[0, 1] == 1:
                    counting_stack[ROILength * paddingHighRes - overlap:ROILength * paddingHighRes + 1, 0:overlap] = 1
                if bdROI[1, 1] == imsize and bdROI[0, 0] == 1:
                    counting_stack[0:overlap, ROILength * paddingHighRes - overlap:ROILength * paddingHighRes + 1] = 1
                himAPIC = himAPIC_Temp[himAPIC_Tempsize[0][0]-1:himAPIC_Tempsize[1][0],himAPIC_Tempsize[0][1]-1: himAPIC_Tempsize[1][1]]
                counting_stack = counting_stack[counting_stack_Tempsize[0][0]-1:counting_stack_Tempsize[1][0],counting_stack_Tempsize[0][1]-1: counting_stack_Tempsize[1][1]]
                himAPIC_stack[(bdROI[0][0]-1)*paddingHighRes:bdROI[1][0]*paddingHighRes,(bdROI[0][1]-1)*paddingHighRes:bdROI[1][1]*paddingHighRes] += himAPIC*counting_stack

        ## Conpensate for the overlapping area
        counting_stack = torch.ones([ROILength * paddingHighRes, ROILength * paddingHighRes], device='cuda')
        overlap = overlapping_size * paddingHighRes
        for j in range(overlap):
            alpha = j / overlap
            counting_stack[:, j] = alpha
            counting_stack[:, -j - 1] = alpha
        for i in range(overlap):
            alpha = i / overlap
            counting_stack[i, :] = torch.min(counting_stack[i, :],
                                             torch.full([ROILength * paddingHighRes], alpha, device='cuda'))
            counting_stack[-i - 1, :] = torch.min(counting_stack[-i - 1, :],
                                                  torch.full([ROILength * paddingHighRes], alpha,
                                                             device='cuda'))
        top_left = counting_stack[:overlap,:overlap]
        top_right = counting_stack[:overlap, -overlap-1:-1]
        bottom_right = counting_stack[-overlap - 1:-1, -overlap - 1:-1]
        bottom_left = counting_stack[-overlap - 1:-1,:overlap]
        conp_mask = top_left + top_right + bottom_right + bottom_left

        for crop_i in range(batch_number-1):
            for crop_j in range(batch_number-1):
                right_bottom_point = [(crop_i+1)*batch_size*3, (crop_j+1)*batch_size*3]
                himAPIC_stack[round(right_bottom_point[0]-overlap/2-1):round(right_bottom_point[0]+overlap/2-1), round(right_bottom_point[1]-overlap/2-1):round(right_bottom_point[1]+overlap/2-1)]/=conp_mask

        new_cmap = plt.get_cmap('seismic').reversed()
        plt.imshow(np.angle(himAPIC_stack.cpu()),cmap = new_cmap)
        plt.clim(-np.pi, np.pi)
        plt.colorbar()
        plt.axis('image')
        plt.axis('off')
        plt.title('Phase, APIC, Whole FOV')
        plt.show()

        himAPIC_Phase = np.angle(himAPIC_stack.cpu().numpy())
        plt.imshow(himAPIC_Phase[3756:3756+1536, 3950:3950+1536],cmap = new_cmap)
        plt.clim(-np.pi, np.pi)
        plt.colorbar()
        plt.axis('image')
        plt.axis('off')
        plt.title('Phase, APIC, Selected FOV for the paper')
        plt.show()
        if saveResult:
            os.makedirs('Results', exist_ok=True)
            file_path = os.path.join('Results', fileName[5:-4] + '_APIC_Results.mat')
            sio.savemat(file_path, {'himAPIC_stack': himAPIC_stack.cpu().numpy()})




if __name__ == "__main__":

    main()
