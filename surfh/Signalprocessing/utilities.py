import numpy as np
from einops import einsum, rearrange


def mad_std(x):
    """Robust standard deviation from Median Absolute Deviation (MAD)."""
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))

def sliding_mad(x):
    """Compute sliding MAD-based std over a 1D array."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    window = max(3, n // 10)
    if window % 2 == 0:
        window += 1
    half = window // 2
    result = np.zeros(n)
    for i in range(n):
        start, end = max(0, i - half), min(n, i + half + 1)
        result[i] = mad_std(x[start:end])
    return result


def partitioning_einops2(cube, di, dj):
    new_cube = rearrange(
        cube, "wl (dx bx) (dy by) -> wl (dx dy) bx by", dx=di, dy=dj
    )
    return new_cube

def concat_M(M):
    nb_blocks, _, nb_subblocks, _ = M.shape
    concat_width = nb_blocks * nb_subblocks
    concat = np.zeros((concat_width, concat_width), dtype=complex)  # always a square
    for l in range(nb_blocks):
        for c in range(nb_blocks):
            concat[
                l * nb_subblocks : (l + 1) * nb_subblocks,
                c * nb_subblocks : (c + 1) * nb_subblocks,
            ] += M[l, c, ...]
    return concat

def split_M(M, split_shape):
    split = np.zeros(split_shape, dtype=complex)
    nb_blocks, _, nb_subblocks, _ = split_shape
    for l in range(nb_blocks):
        for c in range(nb_blocks):
            split[l, c, ...] += M[
                l * nb_subblocks : (l + 1) * nb_subblocks,
                c * nb_subblocks : (c + 1) * nb_subblocks,
            ]
    return split

def make_iHtH_spectro(HtH_freq_spectro):
    inv_hess_freq = np.zeros_like(HtH_freq_spectro, dtype=complex)
    print("HtH_freq_spectro.shape =", HtH_freq_spectro.shape)
    print(HtH_freq_spectro[..., 0, 0].shape)


    H, W = inv_hess_freq.shape[-2:]
    # raise NotImplementedError("This function is deprecated, use make_iHtH_spectro_fast instead.")
    for h in range(H):
        for w in range(W):
            M = np.copy(HtH_freq_spectro[..., h, w])
            C = concat_M(M)
            iC = np.linalg.inv(C)
            S = split_M(iC, inv_hess_freq.shape[:4])
            inv_hess_freq[..., h, w] += S
    return inv_hess_freq

import numpy as np

def make_iHtH_spectro_fast(HtH_freq_spectro, eps=1e-8):
    """
    HtH_freq_spectro: shape (f1, f2, c1, c2, H, W) = (13,13,25,25,25,25)
    Returns inv_hess_freq with same shape.
    eps: small Tikhonov regularization added to diagonal to avoid singulars.
    """
    print("HtH_freq_spectro.shape =", HtH_freq_spectro.shape)
    f1, f2, c1, c2, H, W = HtH_freq_spectro.shape
    assert f1 == f2, "expected f1==f2"
    assert c1 == c2, "expected c1==c2"
    n_rows = f1 * c1   # 13*25 = 325
    n_cols = f2 * c2   # 13*25 = 325
    assert n_rows == n_cols, "matrix must be square"

    # 1) Move spatial axes to front: (H,W,f1,f2,c1,c2)
    X = np.moveaxis(HtH_freq_spectro, (-2, -1), (0, 1))  # (H,W,f1,f2,c1,c2)

    # 2) Rearrange to group row indices (f1,c1) and col indices (f2,c2):
    #    desired order: (H, W, f1, c1, f2, c2)
    X = X.transpose(0, 1, 2, 4, 3, 5)  # (H, W, f1, c1, f2, c2)

    # 3) Flatten to batch of matrices: (batch, n, n) with batch = H*W
    batch = H * W
    X = X.reshape(batch, f1 * c1, f2 * c2)  # (batch, 325, 325)

    # 4) Regularize (Tikhonov) to avoid singular matrices
    n = X.shape[-1]
    I = np.eye(n, dtype=X.dtype)[None, :, :]           # (1,n,n)
    I_batch = np.repeat(I, batch, axis=0)              # (batch,n,n)
    X_reg = X + eps * I_batch                          # broadcast regularization

    # 5) Solve A X = I  (batch): use np.linalg.solve for better stability than inv
    #    np.linalg.solve accepts stacks: solves for each batch A[i] and B[i].
    #    B is identity -> we obtain the inverse.
    # Note: if memory is tight, you can loop over smaller chunks (see notes below).
    inv_batch = np.linalg.solve(X_reg, I_batch)        # (batch, n, n)

    # 6) Un-flatten and put axes back to original order:
    inv_batch = inv_batch.reshape(H, W, f1, c1, f2, c2)    # (H, W, f1, c1, f2, c2)

    # 7) Transpose back to (f1, f2, c1, c2, H, W)
    inv_hess_freq = inv_batch.transpose(2, 4, 3, 5, 0, 1)  # (f1, f2, c1, c2, H, W)

    return inv_hess_freq


import torch
import numpy as np

def make_iHtH_spectro_gpu_torch(HtH_freq_spectro, eps=1e-8, device="cuda"):
    X = torch.from_numpy(HtH_freq_spectro).to(device)

    f1, f2, c1, c2, H, W = X.shape

    X = torch.movedim(X, (-2, -1), (0, 1))
    X = X.permute(0, 1, 2, 4, 3, 5)

    batch = H * W
    n = f1 * c1

    X = X.reshape(batch, n, n)

    I = torch.eye(n, device=device, dtype=X.dtype).unsqueeze(0)
    X_reg = X + eps * I

    inv_batch = torch.linalg.solve(X_reg, I)

    inv_batch = inv_batch.reshape(H, W, f1, c1, f2, c2)
    inv_hess_freq = inv_batch.permute(2, 4, 3, 5, 0, 1)

    return inv_hess_freq.cpu().numpy()


# diff with concatenating: now works with decim different for both dimensions
def concatenating2(cubef, shape_target, di, dj):
    n_maps, d1_times_d2, h_block, w_block = cubef.shape
    h, w = shape_target
    
    concatenated_cube = np.zeros((n_maps, h, w), dtype=complex)
    k = 0
    for i in range(di):
        for j in range(dj):
            concatenated_cube[:, i * h_block : (i+1) * h_block, j * w_block : (j+1) * w_block] += cubef[:, k, :, :]
            k += 1
    
    return concatenated_cube

# input in freq and not part, output in freq
def apply_hessian_freq(hess_spec_freq, di, dj, shape_target, x_freq):
    # partitionnement de x
    part_x_freq = partitioning_einops2(x_freq, di, dj)  # (5, 25, 50, 100)

    # produit de HtH avec x
    HtH_x_freq = hess_spec_freq * part_x_freq[np.newaxis, :, np.newaxis, :, :, :]
    # (5, 5, 25, 25, 50, 100) * (1, 5, 1, 25, 50, 100) = (5, 5, 25, 25, 50, 100)

    HtH_x_freq_sum = einsum(HtH_x_freq, "ti tj di dj h w -> ti di h w")

    # reconstitution des cartes en freq
    concat_HtH_x_freq = concatenating2(HtH_x_freq_sum, shape_target, di, dj)
    # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)

    return concat_HtH_x_freq

