import numpy as np
from surfh.ToolsDir import cython_2D_interpolation
from surfh.ToolsDir import cythons_files



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

"""
Spectral blurring
"""
def wblur(arr: np.ndarray, wpsf: np.ndarray, num_threads: int) -> np.ndarray:
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
    #arr = np.moveaxis(arr, 0, -1)
    result_array = cythons_files.c_wblur(np.ascontiguousarray(arr).astype(np.float64), 
                                         np.ascontiguousarray(wpsf).astype(np.float64), 
                                         wpsf.shape[1], arr.shape[1], 
                                         arr.shape[2], wpsf.shape[0],
                                         num_threads)
    return result_array


def wblur_t(arr: np.ndarray, wpsf: np.ndarray, num_threads: int) -> np.ndarray:
    """Apply transpose of blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ', α, β].
    wpsf: array-like
      Wavelength PSF of shape [λ', λ, β]

    Returns
    -------
    out: array-like
      A wavelength blurred array in [λ, α, β].
    """
    # [λ, α, β] = ∑_λ' arr[λ', α, β] wpsf[λ', λ]
    # Σ_λ'
    result_array = cythons_files.c_wblur_t(arr, wpsf, wpsf.shape[1], 
                                           arr.shape[1], arr.shape[2], 
                                           wpsf.shape[0], num_threads)
    return result_array