"""
Cythonized routines for the RegularGridInterpolator.
""" 

from libc.math cimport NAN
import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free, calloc

ctypedef double complex double_complex


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef find_interval_ascending(double* x,
                            size_t nx,
                            double xval,
                            int prev_interval=0,
                            bint extrapolate=1):
    """
    Find an interval such that x[interval] <= xval < x[interval+1]. Assuming
    that x is sorted in the ascending order.
    If xval < x[0], then interval = 0, if xval > x[-1] then interval = n - 2.

    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.

    """
    # high, low, mid
    cdef:
        int high, low, mid
        int interval = prev_interval
        double a = x[0]
        double b = x[nx - 1]
    if interval < 0 or interval >= nx:
        interval = 0

    # Check if the value to find is in the range of the input value
    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    # Check if the value to find is at eh boundery
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)

        # Split the value to search in half (not equal )
        if xval >= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        # En gros c'est un "Guess the number", en prenant la moitié de l'intervale entre high et low à chaque iteration
        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def find_indices(tuple grid not None, const double[:, :] xi):

    cdef:
        long i, j, grid_i_size
        double denom, value
        # const is required in case grid is read-only
        const double[::1] grid_i

        # Axes to iterate over
        long I = xi.shape[0]
        long J = xi.shape[1]

        int index = 0

        # Indices of relevant edges between which xi are situated
        long[:,::1] indices = np.empty_like(xi, dtype=int)

        # Distances to lower edge in unity units
        double[:,::1] norm_distances = np.zeros_like(xi, dtype=float)

    # iterate through dimensions
    for i in range(I):
        
        grid_i = grid[i] # = array of 512 element
        grid_i_size = grid_i.shape[0] # = 512, the size of the input grid

        for j in range(J):
            value = xi[i, j] # Select one one of the 2D coordinate of xi

            # Return the low idx of the input grid corresponding to the current output Alpha or Beta coordinate
            index = find_interval_ascending(&grid_i[0],
                                            grid_i_size,
                                            value,
                                            prev_interval=index,
                                            extrapolate=1)
            indices[i, j] = index


            # Return the equivalent of "step", the Alpha or Beta distance between 2 input pixels grid
            denom = grid_i[index + 1] - grid_i[index]

            # Compute the distance
            norm_distances[i, j] = (value - grid_i[index]) / denom


    return np.asarray(indices), np.asarray(norm_distances)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def solve_2D_hypercube(const long[:,:] c_indices, const double[:,:]c_norm_distances, 
                       const double[:,:,:]c_in_values, int shape_j, int nWave):

    cdef:
        int num_points = shape_j
        int j = 0
        long i0, i1 = 0
        double y0, y1 = 0.
        double w1, w2, w3, w4 = 0.
       
        double[:,:]c_out_values = np.ascontiguousarray(np.zeros((nWave, num_points)))
   
    for j in range(num_points):
        i0 = c_indices[0, j]
        i1 = c_indices[1, j]

        y0 = c_norm_distances[0,j]
        y1 = c_norm_distances[1,j]

        w1 = ((1. - y0) * (1. - y1))
        w2 = ((1. - y0) * y1)
        w3 = (y0 * (1. - y1))
        w4 = (y0 * y1)

        for l in range(nWave):
            c_out_values[l,j] = c_in_values[l, i0, i1]*w1
            c_out_values[l,j] = c_out_values[l,j] + c_in_values[l, i0, i1+1]*w2
            c_out_values[l,j] = c_out_values[l,j] + c_in_values[l, i0+1, i1]*w3
            c_out_values[l,j] = c_out_values[l,j] + c_in_values[l, i0+1, i1+1]*w4

    return np.asarray(c_out_values)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_wblur(const double[:,:,:] arr, const double[:,:,:]wpsf, 
                int sizeLambda, int sizeAlpha, int sizeBeta,
                int sizeLambdaPrime, int num_threads):

    """Apply blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ, α, β].
    wpsf: array-like
      Wavelength PSF of shape [λ', λ, β]

    Returns
    -------
    out: array-like
      A wavelength blurred array in [λ', α, β].
    """
    # [λ', α, β] = ∑_λ arr[λ, α, β] wpsf[λ', λ, β]
    # Σ_λ
    cdef:
        double[:,:,:] c_res = np.zeros((sizeLambdaPrime, sizeAlpha, sizeBeta))
        int l, a, b, ll = 0
        double tmp = 0.

    with nogil, parallel(num_threads=num_threads):
        for ll in prange(sizeLambdaPrime):
            for a in range(sizeAlpha):
                for b in range(sizeBeta):
                    tmp = 0 
                    for l in range(sizeLambda):
                        tmp = tmp + arr[l,a,b]* wpsf[ll,l,b]
                    c_res[ll,a,b] = tmp 

    return np.asarray(c_res)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_cubeToSlice(const double[:,:,:] arr, const double[:,:,:]dirac, 
                int sizeLambda, int sizeAlpha, int sizeBeta,
                int sizeLambdaPrime, int num_threads):
    """Apply transpose of blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ', α, β].

    Returns
    -------
    c_res: array-like
      A wavelength blurred array in [λ, α, β].
    """
    # [λ, α, β] = ∑_λ' arr[λ', α, β] dirac[λ', λ]
    # Σ_λ'
    cdef:
        double[:,:,:] c_res = np.zeros((sizeLambdaPrime, sizeAlpha, sizeBeta))
        int l, a, b, ll = 0
        double tmp = 0.

    with nogil, parallel(num_threads=num_threads):
        for ll in prange(sizeLambdaPrime):
            for a in range(sizeAlpha):
                for b in range(sizeBeta):
                    tmp = 0 
                    for l in range(sizeLambda):
                        tmp = tmp + arr[l,a,b]* dirac[ll,l,b]
                    c_res[ll,a,b] = tmp 

    for a in range(sizeAlpha):
        for b in range(sizeBeta):
            for l in range(1, sizeLambdaPrime-1):
                if c_res[l,a,b] <1e-2: # Threshold to 1e-2 instead of 0
                    c_res[l,a,b] = (c_res[l-1,a,b] + c_res[l+1,a,b])/2

    return np.asarray(c_res)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_wblur_t(const double[:,:,:] arr, const double[:,:,:]wpsf, 
                int sizeLambda, int sizeAlpha, int sizeBeta,
                int sizeLambdaPrime, int num_threads):
    """Apply transpose of blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ', α, β].
    wpsf: array-like
      Wavelength PSF of shape [λ', λ, β]

    Returns
    -------
    c_res: array-like
      A wavelength blurred array in [λ, α, β].
    """
    # [λ, α, β] = ∑_λ' arr[λ', α, β] wpsf[λ', λ]
    # Σ_λ'
    cdef:
        double[:,:,:] c_res = np.zeros((sizeLambda, sizeAlpha, sizeBeta))
        int l, a, b, ll = 0
        double tmp = 0.

    with nogil, parallel(num_threads=num_threads):
        for l in prange(sizeLambda):
            for a in range(sizeAlpha):
                for b in range(sizeBeta):
                    tmp = 0 
                    for ll in range(sizeLambdaPrime):
                        tmp = tmp + arr[ll,a,b]* wpsf[ll,l,b]
                    c_res[l,a,b] = tmp   
        
    return np.asarray(c_res)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_sliceToCube_t(const double[:,:,:] arr, const double[:,:,:]dirac, 
                int sizeLambda, int sizeAlpha, int sizeBeta,
                int sizeLambdaPrime, int num_threads):
    """Apply transpose of blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ', α, β].

    Returns
    -------
    c_res: array-like
      A wavelength blurred array in [λ, α, β].
    """
    # [λ, α, β] = ∑_λ' arr[λ', α, β] dirac[λ', λ]
    # Σ_λ'
    cdef:
        double[:,:,:] c_res = np.zeros((sizeLambda, sizeAlpha, sizeBeta))
        int l, a, b, ll = 0
        double tmp = 0.

    with nogil, parallel(num_threads=num_threads):
        for l in prange(sizeLambda):
            for a in range(sizeAlpha):
                for b in range(sizeBeta):
                    tmp = 0 
                    for ll in range(sizeLambdaPrime):
                        tmp = tmp + arr[ll,a,b]* dirac[ll,l,b]
                    c_res[l,a,b] = tmp 

    # for a in range(sizeAlpha):
    #     for b in range(sizeBeta):
    #         for l in range(1, sizeLambda-1):
    #             if c_res[l,a,b] <1e-2: # Threshold to 1e-2 instead of 0
    #                 c_res[l,a,b] = (c_res[l-1,a,b] + c_res[l+1,a,b])/2

    return np.asarray(c_res)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_precompute_TST(int Nmaps, int Nx, int Ny,
                   int Nlam,
                   float[:,:,:] S,
                   float[:,:] templates):
    cdef:
        float[:,:,:,:] TST = np.zeros((Nmaps, Nmaps, Nx, Ny), dtype=np.float32)
        int m, mp, lam, i, j = 0
        float tmp1, tmp2 = 0.

    for m in range(Nmaps):
        for mp in range(Nmaps):
            for i in range(Nx):
                for j in range(Ny):
                    for lam in range(Nlam):
                        tmp1 = templates[mp, lam]
                        tmp2 = templates[m,lam]
                        TST[m, mp, i, j] += tmp1 * tmp2 * S[lam, i, j]
    return TST


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_fast_forward_TST(int NLambda, int Nmaps, int Nx, int Ny,
                       int[:,:] select_arr, int size_select,
                       float[:,:] templates,
                       float[:,:,:] maps):

    cdef:
        float[:,:,:] cube = np.zeros((NLambda, Nx, Ny), dtype=np.float32)
        int m, pix, lam, i, j = 0
    
    with nogil, parallel():
        for pix in prange(size_select):
            lam = select_arr[pix, 0]
            i   = select_arr[pix, 1]
            j   = select_arr[pix, 2]
            for m in range(Nmaps):
                cube[lam, i, j] += maps[m, i, j]*templates[m, lam]
    return cube


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_fast_LMM_maps2cube(int NLambda, int Nmaps, int Nx, int Ny,
                       float[:,:] templates,
                       float[:,:,:] maps):
    cdef:
        float[:,:,:] cube = np.zeros((NLambda, Nx, Ny), dtype=np.float32)
        int m, lam, i, j = 0
        float tmp = 0.
    
    with nogil, parallel():
        for lam in prange(NLambda):
            for i in range(Nx):
                for j in range(Ny):
                    for m in range(Nmaps):
                        tmp += maps[m, i, j]*templates[m, lam]
                    cube[lam, i, j] = tmp
                    tmp = 0 
    return cube


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_fast_adjoint_TST(int NLambda, int Nmaps, int Nx, int Ny,
                       int[:,:] select_arr, int size_select,
                       float[:,:] templates,
                       float[:,:,:] cube):
    cdef:
        float[:,:,:] maps = np.zeros((Nmaps, Nx, Ny), dtype=np.float32)
        int m, pix, lam, i, j = 0

    
    for m in range(Nmaps):        
        for pix in range(size_select):
            lam = select_arr[pix, 0]
            i   = select_arr[pix, 1]
            j   = select_arr[pix, 2]
            maps[m, i, j] = maps[m, i, j] + cube[lam, i, j]*templates[m, lam]

    return maps


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def c_fast_LMM_cube2maps(int NLambda, int Nmaps, int Nx, int Ny,
                       float[:,:] templates,
                       float[:,:,:] cube):
    cdef:
        float[:,:,:] maps = np.zeros((Nmaps, Nx, Ny), dtype=np.float32)
        int m, lam, i, j = 0
        float tmp = 0.
    
    with nogil, parallel():
        for m in range(Nmaps):
            for lam in prange(NLambda):
                for i in range(Nx):
                    for j in range(Ny):
                        maps[m, i, j] += cube[i, j, lam]*templates[m, lam]
                    # maps[m, i, j] = tmp
                    # tmp = 0 
    return maps