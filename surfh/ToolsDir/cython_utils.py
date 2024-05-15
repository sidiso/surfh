import numpy as np
from surfh.ToolsDir import cython_2D_interpolation


"""
Interpolation
"""
def interpn_cube2local(wavel_index, alpha_axis, beta_axis, cube, local_coords, local_shape):
    """
    Interpolate hyperspectral cube coordinates onto local FoV coordinates.
    Interpolation - cube -> FoV

    Parameters:
    ----------
    wavel_index: array-like
      Index array of the wavelength vector
    alpha_axis: array-like
      Alpha coordinates of the image cube, 1D array
    beta_axis: array-like
      Beta coordinates of the image cube, 1D array
    cube: array-like
      Hyperspectral cube
    local_coords: array-like
      list of 2D points of each local coordinates in the global coordinate system
    local_shape: Tuple(int, int, int)
      3D shape of the local hyperspectral cube 
    """
    return cython_2D_interpolation.interpn( (alpha_axis, beta_axis), cube, local_coords, len(wavel_index)).reshape(local_shape)


def interpn_local2cube(wavel_index, local_alpha_axis, local_beta_axis, cube, global_coords, global_shape):
    """
    Interpolate hyperspectral cube coordinates onto local FoV coordinates.
    Interpolation - cube -> FoV

    Parameters:
    ----------
    wavel_index: array-like
      Index array of the wavelength vector
    alpha_axis: array-like
      Alpha coordinates of the image cube, 1D array
    beta_axis: array-like
      Beta coordinates of the image cube, 1D array
    cube: array-like
      Hyperspectral cube
    local_coords: array-like
      list of 2D points of each local coordinates in the global coordinate system
    local_shape: Tuple(int, int, int)
      3D shape of the local hyperspectral cube 
    """
    return cython_2D_interpolation.interpn( (local_alpha_axis, local_beta_axis), 
                                              cube, 
                                              global_coords, 
                                              len(wavel_index),
                                              bounds_error=False, 
                                              fill_value=0,).reshape(global_shape)

