from numba import njit

@njit(fastmath=True)
def apply_slit_slice(local_cube, slit, w, ii, jj):
    """
    local_cube: (L, H, W)
    slit:       (L, N)
    w:          (L, N)
    ii, jj:     (N,)
    """

    L = slit.shape[0]
    N = ii.size

    for l in range(L):
        for k in range(N):
            local_cube[l, ii[k], jj[k]] += slit[l, k] * w[l, k]


from numba import njit, prange
import numpy as np

@njit(parallel=True)
def slicing_t_numba(local_cube, slit, ii, jj, weights):
    """
    local_cube: array à remplir, shape (L, alpha, beta)
    slit: array de la slit, shape (L, len(ii), len(jj))
    ii, jj: arrays d'indices des slices
    weights: array broadcastable sur slit, shape (1, len(ii), len(jj))
    """
    L = slit.shape[0]

    # boucle sur la longueur
    for l in prange(L):
        for i_idx in range(len(ii)):
            i = ii[i_idx]
            for j_idx in range(len(jj)):
                j = jj[j_idx]
                local_cube[l, i, j] += slit[l, i_idx, j_idx] * weights[0, i_idx, j_idx]
