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



@njit(parallel=True)
def slicing_numba(gridded_cube, ii, jj, weights):
    """
    gridded_cube : (L, H, W)
    ii, jj       : (N,) indices
    weights      : (L, N)
    return       : (L, N)
    """
    L = gridded_cube.shape[0]
    Ni = ii.shape[0]
    Nj = jj.shape[0]

    out = np.zeros((L, Ni, Nj), dtype=gridded_cube.dtype)    
    
    for l in prange(L):              # parallèle sur lambda
        for k in range(Ni):          # index dans ii
            i = ii[k]
            for m in range(Nj):      # index dans jj
                j = jj[m]
                out[l, k, m] = gridded_cube[l, i, j] * weights[0, k, m]                
    return out

@njit(parallel=True)
def slicing_numba_prealloc(gridded_cube, ii, jj, weights, out):
    """
    out : (L, N) array déjà alloué
    """
    L = gridded_cube.shape[0]
    N = ii.shape[0]

    for l in prange(L):
        for k in range(N):
            out[l, k] = gridded_cube[l, ii[k], jj[k]] * weights[0, l, k]

    return out
