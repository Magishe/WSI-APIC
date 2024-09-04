import numpy as np
import torch
def exact_shift(im, relative_pixel, **kwargs):
    """
    Function to perform sub-pixel shift on an image.
    im: Input image.
    relative_pixel: Shift amount in pixels.
    isRealSpace: Flag to indicate if operation is in real or Fourier space.
    output_size: 'same' to keep the output image size as input, 'padded' to pad the image.
    """

    [xsize, ysize] = [im.shape[0],im.shape[1]]
    # Default settings
    sameSize = True
    isRealSpace = True

    # Parsing kwargs
    for key, value in kwargs.items():
        key = key.lower()
        if key in ['isrealspace', 'realspace']:
            if value:
                isRealSpace = True
            else:
                isRealSpace = False
        elif key == 'size':
            value = value.lower()
            if value == 'same':
                sameSize = True
            elif value == 'padded':
                sameSize = False
        else:
            raise ValueError('Option is not supported.')

    # Adjusting image size if sameSize is True
    if sameSize:
        o_xsize = xsize
        o_ysize = ysize

    # Padding the image if necessary
    if xsize % 2 == 0:
        im = torch.nn.functional.pad(im, (0, 0, 0, 1), mode='constant')
    if ysize % 2 == 0:
        im = torch.nn.functional.pad(im, (0, 1, 0, 0), mode='constant')
    if im.dim() == 2:
        [xsize, ysize] = im.shape
        im = im.unsqueeze(2)
        num = 1
    elif im.dim() == 3:
        [xsize, ysize,num] = im.shape
    else:
        raise ValueError('Only support 2D or 3D images.')
    if np.ndim(relative_pixel) == 1:
        relative_pixel = relative_pixel.reshape((1, 2))
    # Adjusting relative_pixel values
    relative_pixel[:, 0] = relative_pixel[:, 0] - xsize * np.fix(relative_pixel[:, 0]/xsize)
    relative_pixel[:, 1] = relative_pixel[:, 1] - ysize * np.fix(relative_pixel[:, 1]/ysize)

    # Creating meshgrid
    X, Y = torch.meshgrid(torch.arange(1, ysize + 1, device='cuda'),torch.arange(1, xsize + 1, device='cuda'),indexing = 'ij')
    xc = np.floor(xsize / 2 + 1)
    yc = np.floor(ysize / 2 + 1)

    # Adjusting coordinates based on real or Fourier space
    if isRealSpace:
        yr = Y - yc
        xr = X - xc
    else:
        yr = Y - 1
        xr = X - 1

    # Calculating shift magnitude and angle
    r_shift = np.sqrt(relative_pixel[:, 0]**2 + relative_pixel[:, 1]**2)
    shift_angle = np.zeros(num)
    f_r = np.zeros((num, 2))

    for ii in range(num):
        if relative_pixel[ii, 1] == 0:
            shift_angle[ii] = np.pi / 2 * np.sign(relative_pixel[ii, 0])
        else:
            shift_angle[ii] = np.arctan(relative_pixel[ii, 0] / relative_pixel[ii, 1])
            if relative_pixel[ii, 1] < 0:
                shift_angle[ii] -= np.pi

        if r_shift[ii] != 0:
            f_r[ii, 0] = xsize / r_shift[ii]
            f_r[ii, 1] = ysize / r_shift[ii]


    final_xsize = o_xsize if sameSize else xsize
    final_ysize = o_ysize if sameSize else ysize

    shift_im = torch.zeros(final_xsize, final_ysize, num, dtype=im.dtype).cuda()

    for ii in range(num):
        fr_temp = f_r[ii, :]
        if not np.all(fr_temp == 0):
            my_angle = torch.tensor(shift_angle[ii])
            if isRealSpace:
                ft = torch.fft.fftshift(torch.fft.fft2(im[:, :, ii], dim=[-2, -1]), dim=[-2, -1])
                ft *= torch.exp(-1j * 2 * torch.pi * (xr * torch.sin(my_angle) / fr_temp[0] + yr * torch.cos(my_angle) / fr_temp[1]))
                temp = torch.fft.ifft2(torch.fft.ifftshift(ft, dim=[-2, -1]), dim=[-2, -1])
            else:
                ft = torch.fft.ifftshift(torch.fft.ifft2(im[:, :, ii], dim=[-2, -1]), dim=[-2, -1])
                ft *= torch.exp(1j * 2 * torch.pi * (xr * torch.sin(my_angle) / fr_temp[0] + yr * torch.cos(my_angle) / fr_temp[1]))
                temp = torch.fft.fftshift(torch.fft.fft2(ft, dim=[-2, -1]), dim=[-2, -1])
            shift_im[:, :, ii] = torch.real(temp[:final_xsize, :final_ysize])  # In fact temp is real. The imaginary part is so small.
        else:
            shift_im[:, :, ii] = im[:final_xsize, :final_ysize, ii]
    if num == 1:
        shift_im = shift_im.squeeze(2)
    return shift_im
