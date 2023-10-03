"""
Cythonized routines for the RegularGridInterpolator.
""" 

from libc.math cimport NAN
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free, calloc

ctypedef double complex double_complex




cpdef dist(point1, point2):
    x = (point1[0] - point2[0]) ** 2
    y = (point1[1] - point2[1]) ** 2
    return np.sqrt(x + y)


cpdef test(point1, point2):
    x = point1 + point2 + point1 + point1
    return x


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
def element_wise_vector_multiplication(a, b, size):

    cdef:
        double[::1] c_a = a
        double[::1] c_b = b
        double[::1] res = a
        int c_size = size
        int i = 0

    for i in range(c_size):
        res[i] = c_a[i] * c_b[i]

    return res


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double c_element_wise_vector_multiplication(double* c_a, double* c_b, int c_size, double *res):

    cdef:
        int i = 0

    for i in range(c_size):
        res[i] = c_a[i] * c_b[i]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def solve_hypercube(const long[:] c_all_indices, const double[:]c_all_norm_distances, const double[:]c_in_values, shape_i, shape_j, image_size):

    cdef:
        int size_i = shape_i
        int size_j = shape_j
        int i,j = 0
        int id0, id1, id2 = 0
        
        #long[::1]c_all_indices = np.ascontiguousarray(all_indices) # vector
        #double[::1]c_in_values = np.ascontiguousarray(in_values_ravel)
        double[::1]c_out_values = np.ascontiguousarray(np.zeros(shape_j))
        #double[::1]c_all_norm_distances = np.ascontiguousarray(all_norm_distances) # Vector



        #double *c_weight = <double *>c_valsmalloc(size_j * sizeof(double))
        #double *c_vals = <double *>malloc(size_j * sizeof(double))
        double[::1]c_weight = np.ascontiguousarray(np.zeros(shape_j))
        double[::1]c_vals = np.ascontiguousarray(np.zeros(shape_j))

        double[::1]term = np.ascontiguousarray(np.zeros(shape_j))
        #double *term = <double *>malloc(size_j * sizeof(double))


        
        cdef int c_1D_size = image_size
        cdef int c_2D_size = c_1D_size*c_1D_size
    

    for i in reversed(range(8)):
        id0 = (i&0b001) >> 0
        id1 = (i&0b010) >> 1
        id2 = (i&0b100) >> 2

        for j in range(size_j): # 1xxx dim
            c_weight[j] = c_all_norm_distances[id0*3*size_j + j] *\
                        c_all_norm_distances[id1*3*size_j + size_j + j] *\
                        c_all_norm_distances[id2*3*size_j + 2*size_j + j]


        for j in range(size_j):
            idx =   c_all_indices[id0*3*size_j + j]*c_2D_size +\
                    c_all_indices[id1*3*size_j + size_j + j]*c_1D_size +\
                    c_all_indices[id2*3*size_j + 2*size_j + j]
            c_vals[j] = c_in_values[idx]

        c_element_wise_vector_multiplication(&c_vals[0], &c_weight[0], size_j, &term[0])
        for j in range(size_j):
            c_out_values[j] += term[j]

    #free(term)
    #free(c_vals)
    #free(c_weight)

    return np.asarray(c_out_values)



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