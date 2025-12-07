import numpy as np
from numba import njit, prange


"""
Methods for adjoint interpolations
"""
@njit
def precompute_gridding_t_indices(local_alpha, local_beta, alpha_coords, beta_coords):
    npts = len(alpha_coords)

    i0 = np.empty(npts, dtype=np.int64)
    j0 = np.empty(npts, dtype=np.int64)
    i1 = np.empty(npts, dtype=np.int64)
    j1 = np.empty(npts, dtype=np.int64)

    wa0 = np.empty(npts, dtype=np.float64)
    wa1 = np.empty(npts, dtype=np.float64)
    wb0 = np.empty(npts, dtype=np.float64)
    wb1 = np.empty(npts, dtype=np.float64)

    for k in range(npts):
        a = alpha_coords[k]
        b = beta_coords[k]

        # searchsorted à la main (numba-friendly)
        ii0 = 0
        while ii0 < len(local_alpha) - 1 and local_alpha[ii0+1] <= a:
            ii0 += 1

        jj0 = 0
        while jj0 < len(local_beta) - 1 and local_beta[jj0+1] <= b:
            jj0 += 1

        if ii0 >= len(local_alpha)-1:
            ii0 = len(local_alpha)-2
        if jj0 >= len(local_beta)-1:
            jj0 = len(local_beta)-2

        i0[k] = ii0
        j0[k] = jj0

        i1[k] = ii0 + 1
        j1[k] = jj0 + 1

        # Poids en alpha
        denom_a = (local_alpha[ii0+1] - local_alpha[ii0])
        if denom_a == 0:
            wa1[k] = 0.0
        else:
            wa1[k] = (a - local_alpha[ii0]) / denom_a
        wa0[k] = 1.0 - wa1[k]

        # Poids en beta
        denom_b = (local_beta[jj0+1] - local_beta[jj0])
        if denom_b == 0:
            wb1[k] = 0.0
        else:
            wb1[k] = (b - local_beta[jj0]) / denom_b
        wb0[k] = 1.0 - wb1[k]

    return i0, i1, j0, j1, wa0, wa1, wb0, wb1


@njit(fastmath=True)
def gridding_t_slice(slice2d, i0, i1, j0, j1, wa0, wa1, wb0, wb1):
    npts = len(i0)
    out = np.empty(npts, dtype=np.float64)

    for k in range(npts):  # simple loop, très rapide
        f00 = slice2d[i0[k], j0[k]]
        f01 = slice2d[i0[k], j1[k]]
        f10 = slice2d[i1[k], j0[k]]
        f11 = slice2d[i1[k], j1[k]]

        out[k] = (
            wa0[k] * (wb0[k] * f00 + wb1[k] * f01)
            + wa1[k] * (wb0[k] * f10 + wb1[k] * f11)
        )
    return out

@njit(parallel=True, fastmath=True)
def gridding_t_cube(local_cube,
                    i0, i1, j0, j1,
                    wa0, wa1, wb0, wb1):

    nl, nx, ny = local_cube.shape
    npts = len(i0)
    out = np.empty((nl, npts), dtype=np.float64)

    for z in prange(nl):   # ← parallèle sur λ
        out[z] = gridding_t_slice(
            local_cube[z], 
            i0, i1, j0, j1,
            wa0, wa1, wb0, wb1
        )

    return out



"""
Methods for Forward interpolation
"""
@njit
def precompute_gridding_indices(alpha_axis, beta_axis, alpha_coords, beta_coords):
    """
    Pré-calcul pour interpolation 2D sur la grille alpha_axis × beta_axis
    pour des coordonnées alpha_coords, beta_coords (aplaties).
    """
    npts = len(alpha_coords)

    i0 = np.empty(npts, dtype=np.int64)
    j0 = np.empty(npts, dtype=np.int64)
    i1 = np.empty(npts, dtype=np.int64)
    j1 = np.empty(npts, dtype=np.int64)

    wa0 = np.empty(npts, dtype=np.float64)
    wa1 = np.empty(npts, dtype=np.float64)
    wb0 = np.empty(npts, dtype=np.float64)
    wb1 = np.empty(npts, dtype=np.float64)

    for k in range(npts):
        a = alpha_coords[k]
        b = beta_coords[k]

        # searchsorted manuel (compatible numba)
        ii0 = 0
        while ii0 < len(alpha_axis)-1 and alpha_axis[ii0+1] <= a:
            ii0 += 1
        jj0 = 0
        while jj0 < len(beta_axis)-1 and beta_axis[jj0+1] <= b:
            jj0 += 1

        # clamp
        if ii0 >= len(alpha_axis)-1:
            ii0 = len(alpha_axis)-2
        if jj0 >= len(beta_axis)-1:
            jj0 = len(beta_axis)-2

        i0[k] = ii0
        j0[k] = jj0
        i1[k] = ii0 + 1
        j1[k] = jj0 + 1

        # poids en alpha
        da = alpha_axis[ii0+1] - alpha_axis[ii0]
        wa1[k] = 0.0 if da == 0 else (a - alpha_axis[ii0]) / da
        wa0[k] = 1.0 - wa1[k]

        # poids en beta
        db = beta_axis[jj0+1] - beta_axis[jj0]
        wb1[k] = 0.0 if db == 0 else (b - beta_axis[jj0]) / db
        wb0[k] = 1.0 - wb1[k]

    return i0, i1, j0, j1, wa0, wa1, wb0, wb1


@njit(fastmath=True)
def gridding_slices(slice2d, i0, i1, j0, j1, wa0, wa1, wb0, wb1):
    npts = len(i0)
    out = np.empty(npts, dtype=np.float64)

    for k in range(npts):
        f00 = slice2d[i0[k], j0[k]]
        f01 = slice2d[i0[k], j1[k]]
        f10 = slice2d[i1[k], j0[k]]
        f11 = slice2d[i1[k], j1[k]]

        out[k] = (
            wa0[k] * (wb0[k] * f00 + wb1[k] * f01)
          + wa1[k] * (wb0[k] * f10 + wb1[k] * f11)
        )

    return out

@njit(parallel=True, fastmath=True)
def gridding_cube(cube, 
                              i0, i1, j0, j1,
                              wa0, wa1, wb0, wb1):

    nl, nx, ny = cube.shape
    npts = len(i0)
    out = np.empty((nl, npts), dtype=np.float64)

    for z in prange(nl):  # parallelisation sur les longueurs d’onde
        out[z] = gridding_slices(
            cube[z],
            i0, i1, j0, j1,
            wa0, wa1, wb0, wb1
        )

    return out
