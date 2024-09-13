"""
Cythonized routines for the Complex Modified-Shepard interpolation.
""" 

from libc.math cimport NAN
from libc.math cimport sqrt, exp
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
cdef pixel_distance(float p1_a, float p1_b, 
                   float p2_a, float p2_b,
                   float inv_alpha_res, float inv_lambda_res):
    """
    Compute the pixel distance between two points, scaled according to the resolution of alpha and lambda.
    
    Parameters:
    p1, p2: The two points between which the distance is computed (each is a tuple of (alpha, lambda)).
    inv_alpha_res: Inverse of the resolution (pixel size) in the alpha direction.
    inv_lambda_res: Inverse of the resolution (pixel size) in the lambda direction.
    
    Returns:
    The distance in pixel units.
    """
    cdef:
        float dist1 = 0.
        float dist2 = 0.
        float dist1_square = 0.
        float dist2_square = 0.
        float dist = 0.

    dist1 = (p1_a - p2_a) * inv_alpha_res
    dist2 = (p1_b - p2_b) * inv_lambda_res

    dist1_square = dist1**2
    dist2_square = dist2**2

    dist = float(sqrt(dist1_square + dist2_square))

    return dist

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef exponential_weight(float dist, float p=2, float alpha=2.0):
    """
    Compute the exponential weight for a given distance.
    
    Parameters:
    dist: The pixel distance between points.
    p: The power to raise the distance to (typically 2 for inverse distance).
    alpha: The exponential factor that controls the steepness.
    
    Returns:
    The weight associated with the distance.
    """
    cdef:
        float value = 0.
        float exponent = 0.
    value = -alpha * dist**p
    exponent = exp(value)
    return exponent

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def exponential_modified_shepard(float[:] alpha_coord, float[:] lambda_coord, 
                                 float[:] values, float[:,:] alpha_mesh, 
                                 float[:,:] lambda_mesh, float p=2., 
                                 float alpha=2.0, float pixel_cutoff=1, 
                                 float alpha_res=1.0, float lambda_res=1.0, 
                                 float epsilon=1e-6):
    """
    Apply the Exponential Modified-Shepard interpolation method over a grid.
    
    Parameters:
    points: Tuple of (alpha, lambda) valid coordinates.
    values: Values at the known data points (intensity).
    query_points: Tuple of meshgrid (alpha_mesh, lambda_mesh) for interpolation.
    p: The power for inverse distance weighting (default is 2).
    alpha: The exponential decay factor (default is 2.0).
    pixel_cutoff: The cutoff radius in pixels for influence. Points further than this cutoff will not be considered.
    alpha_res: The resolution in the alpha direction (e.g., pixel width).
    lambda_res: The resolution in the lambda direction (e.g., pixel height).
    epsilon: A small value to avoid division by zero.
    
    Returns:
    Interpolated intensity values for the entire grid.
    """
    cdef:
        int i, j, k
        int nb_values = len(values)
        float inv_alpha_res = 0.
        float inv_lambda_res = 0.
        float numerator = 0.0
        float denominator = 0.0
        float c_epsilon = epsilon
        float w = 0.
        float[:,:] interpolated_values = np.zeros_like(alpha_mesh)

    inv_alpha_res = 1/alpha_res
    inv_lambda_res = 1/lambda_res

    # Iterate over each point in the grid
    for i in range(alpha_mesh.shape[0]):
        for j in range(alpha_mesh.shape[1]):
            # query_point = (alpha_mesh[i, j], lambda_mesh[i, j])
            
            # Initialize numerator and denominator for Shepard's method
            numerator = 0.0
            denominator = 0.0
            
            # Iterate over known points and compute weights
            for k in range(nb_values):
                # pt = (points[0][k], points[1][k])
                dist = pixel_distance(alpha_coord[k], lambda_coord[k], 
                                      alpha_mesh[i, j], lambda_mesh[i, j], 
                                      inv_alpha_res, inv_lambda_res) + c_epsilon  # Avoid division by zero
                
                # Apply pixel cutoff if specified
                if dist <= pixel_cutoff:
                    w = exponential_weight(dist, p=p, alpha=alpha)
                    
                    numerator += w * values[k]
                    denominator += w
            
            # Store interpolated value at this grid point
            interpolated_values[i, j] = numerator / denominator if denominator != 0 else 0.0
    
    return interpolated_values
