import numpy as np
from subfunctionAPIC.calCoord import calCoord

def F(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def iF(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

# 定义对数幅度函数
def logamp(x):
    return np.log10(np.abs(x) + 1)


def preprocessing(NA_seq,imlow_BF_IN,na_cal,patch_size, dark_th, darkframe, num_matched,exposure_DF, exposure_Ring, lambda_g, dpix_c, mag, NA, flag_process, flag_HPR, flag_show,):
    """
    Perform preprocessing for dark noise and hot pixel removal, intensity normalization
    """
    NAx_vis = -NA_seq[:, 1]
    NAy_vis = -NA_seq[:, 0]
    M, N = imlow_BF_IN.shape[:2]
    Nled = imlow_BF_IN.shape[2]

    FOV_c = [np.ceil((M + 1) / 2), np.ceil((N + 1) / 2)]

    patch_x = 0
    patch_y = 0

    ROI_c = [FOV_c[0] - patch_x * patch_size, FOV_c[1] - patch_y * patch_size]


    x_lim = np.array(range(round(ROI_c[0]) - patch_size // 2, round(ROI_c[0]) + patch_size // 2))-1
    y_lim = np.array(range(round(ROI_c[1]) - patch_size // 2, round(ROI_c[1]) + patch_size // 2))-1

    imlow = imlow_BF_IN
    if flag_process:
        I_low = np.zeros((len(x_lim), len(y_lim), Nled), dtype=imlow.dtype)
        hotpixel = darkframe > dark_th
        th = np.mean(darkframe[~hotpixel])  # Threshold of dark noise
        hotpixel_patch = hotpixel[np.ix_(x_lim, y_lim)]
        x_hot, y_hot = np.where(hotpixel_patch)
        for k in range(Nled):
            I_patch = imlow[np.ix_(x_lim, y_lim, [k])][:,:,0]
            Itmp = I_patch
            # Hot pixel removal
            if flag_HPR:
                for x, y in zip(x_hot, y_hot):
                    xlim = [max(x - 1, 0), min(x + 2, Itmp.shape[0])]
                    ylim = [max(y - 1, 0), min(y + 2, Itmp.shape[1])]
                    I_used = Itmp[slice(*xlim), slice(*ylim)]
                    hotpixel_used = hotpixel_patch[slice(*xlim), slice(*ylim)]
                    non_hot_pixel_list = I_used[~hotpixel_used]
                    Itmp[x, y] = np.mean(non_hot_pixel_list) if len(non_hot_pixel_list) > 0 else Itmp[x, y]
            # Save ROI patch images
            I_low[:, :, k] = Itmp
    else:
        I_low = imlow[np.ix_(x_lim, y_lim)]

    # Intensity Normalization
    I_low = I_low.astype(float)
    for k in range(num_matched):
        I_low[:, :, k] = I_low[:, :, k] * exposure_DF / exposure_Ring
    I_low_Ring = I_low[:, :, :num_matched]
    mean_list = np.mean(I_low_Ring, axis=(0, 1))
    mean_all = np.mean(I_low_Ring)
    # Normalize each slice by its mean intensity to match the overall mean intensity
    for k in range(num_matched):
        I_low[:, :, k] = I_low[:, :, k] / mean_list[k] * mean_all


    # Plot the calibration results
    if flag_show:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        if patch_size % 2 == 1:
            du = 1 / (dpix_c / mag * (patch_size - 1))
        else:
            du = 1 / (patch_size * dpix_c / mag)

        idx_u = np.round((-NAx_vis / lambda_g) / du) + np.floor(patch_size / 2)
        idx_v = np.round((-NAy_vis / lambda_g) / du) + np.floor(patch_size / 2)
        idx_r = np.round((na_cal / lambda_g) / du)

        plt.ion()
        for k in range(12):
            plt.figure(figsize=(6, 6))
            ax = plt.gca()


            transformed_image = F(I_low[:, :, k])
            log_amplitude_image = logamp(transformed_image)

            im = ax.imshow(log_amplitude_image, cmap='winter')
            plt.colorbar(im)
            plt.title(f'LED {k + 1}')

            circle = Circle((idx_u[k], idx_v[k]), idx_r, color='red', fill=False, linestyle='--')
            ax.add_patch(circle)

            plt.draw()
            plt.pause(0.5)

        plt.ioff()
        plt.show()

    na_design = np.array([NAx_vis, NAy_vis]).T
    # Calling the calCoord function with the specified parameters
    freqXY, con, _, _, _, _, _, _ = calCoord(na_design / lambda_g, [patch_size, patch_size], dpix_c, mag, NA,
                                             lambda_g)

    # Recalibrating na_rp_cal and freqXY_calib based on the outputs
    na_rp_cal = na_cal / lambda_g * con
    freqXY_calib = freqXY
    na_calib = na_design

    return I_low, na_calib, na_cal, freqXY_calib,na_rp_cal




