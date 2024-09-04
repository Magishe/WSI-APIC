import torch

def toeplitz_torch(c, r):
    """
    Create a Toeplitz matrix in PyTorch.
    c - First column of the Toeplitz matrix as a 1D tensor.
    r - First row of the Toeplitz matrix as a 1D tensor.
    """
    # Ensuring c and r are 1D tensors
    c = c.flatten()
    r = r.flatten()

    # Lengths
    n = c.size(0)
    m = r.size(0)

    # Create a large matrix that includes all elements of c and r
    # This matrix will be larger than the final Toeplitz matrix
    # We will select appropriate elements from this matrix to form the Toeplitz matrix
    large_matrix = torch.cat((c.flip(0), r[1:])).unsqueeze(0)

    # Use stride trick to select elements for the Toeplitz matrix
    indices = torch.arange(n).unsqueeze(1) + torch.arange(m).unsqueeze(0)

    # Ensure indices are within bounds of the large_matrix
    indices = indices.clamp(max=large_matrix.size(1) - 1)
    data_type_str = str(indices.dtype)

    # Index into large_matrix to create Toeplitz matrix
    toeplitz_matrix = large_matrix[:, indices]
    toeplitz_matrix = toeplitz_matrix.squeeze()
    toeplitz_matrix = toeplitz_matrix.flip(0)

    return toeplitz_matrix

def calConvMtx_GPU(ftKnown_input, maskMeas_input, maskUnknown_input):
    """
    Genrate the convolution matrix (in a faster way, and works with sparse entires/measurements)
    The first argument should be the Fourier transform of the known part,
    the second (logic array) should indicate measurement in the same domain
    as the first input, and the third one should indicate the unknown entries
    and, again, in the same domain as the first input.
    """

    Meas_Sum = torch.sum(maskMeas_input)
    Unko_Sum = torch.sum(maskUnknown_input)
    ftKnown = ftKnown_input
    maskMeas = maskMeas_input
    maskUnknown = maskUnknown_input
    xsize, ysize = ftKnown.shape
    if xsize != ysize:
        raise ValueError("Function 'calConvMtx' only works with square images/matrices.")
    n = xsize
    nConv = 2 * n - 1

    temp = torch.zeros((nConv, n),dtype = torch.complex64).cuda()
    temp[:n, :n] = ftKnown
    temp2 = torch.zeros((1, n),dtype = torch.complex64).cuda()
    cellMat2 = torch.zeros((nConv, n, n + 1),dtype = torch.complex64, device='cuda')
    temp2[0, 0] = temp[0, -1]
    for idx in range(n):
        cellMat2[:, :, idx] = toeplitz_torch(temp[:, idx], temp2[0, :])
    convMatrix = torch.zeros((Meas_Sum, Unko_Sum), dtype=torch.complex64, device='cuda')
    first_row = torch.cat((torch.arange(1, n + 1, device='cuda'), torch.full((n - 1,), n + 1, device='cuda')), 0)
    second_row = torch.cat((torch.tensor([1]), torch.full((n - 1,), n + 1)), 0).cuda()

    idxMat = toeplitz_torch(first_row, second_row)

    num2fillLeft = torch.sum(maskMeas, dim=0)
    num2fillRight = torch.sum(maskUnknown, dim=0)
    nonEmptyLeft = torch.any(maskMeas, dim=0)
    rows, cols = (maskUnknown.nonzero(as_tuple=True))
    linear_indices = rows  + cols* maskUnknown.shape[0]
    idxUnknown = torch.sort(linear_indices).values
    sizeMeasX, _ = maskMeas.shape
    sizeUnknownX, _ = maskUnknown.shape
    idxNonEmpty = (num2fillRight != 0)
    cumFillUnknown = torch.cumsum(num2fillRight[idxNonEmpty], dim=0)
    idx2useDim2 = idxUnknown % sizeUnknownX
    idx2useDim3_trace = torch.zeros(len(idxUnknown) + 1, dtype=torch.int32).cuda()  # initialize variable for calculating the indices for the 3rd dim
    idx2useDim3_trace[cumFillUnknown] = 1
    B = (idx2useDim2.unsqueeze(0) * sizeMeasX)
    cellMat2_flattened = cellMat2.permute(2, 1, 0).contiguous().view(-1)
    idxPrevLeft = 0
    for idx1 in range(nConv):
        if nonEmptyLeft[idx1]:
            idx2useDimTemp = idxMat[idx1, :]
            idxLib = idx2useDimTemp[idxNonEmpty]

            idx2useDim3 = idxLib[torch.cumsum(idx2useDim3_trace[:-1],dim=0)]
            A = torch.where(maskMeas[:, idx1])[0].unsqueeze(-1)
            C = ((idx2useDim3 - 1) * sizeMeasX * sizeUnknownX).unsqueeze(0)
            linearInd = A + B + C

            linearInd_flattened = linearInd.view(-1)
            selected_elements = torch.index_select(cellMat2_flattened, 0, linearInd_flattened)
            result = selected_elements.view(*linearInd.shape)
            convMatrix[idxPrevLeft:idxPrevLeft + num2fillLeft[idx1], :] = result
        idxPrevLeft += num2fillLeft[idx1]

    return convMatrix