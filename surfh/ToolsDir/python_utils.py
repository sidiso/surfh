import numpy as np
import udft
import scipy as sp

from typing import List, Tuple


"""
LMM
"""
def lmm_maps2cube(maps: np.ndarray, tpls: np.ndarray) -> np.ndarray:
    """
    Apply the Linear Mixing Model transform from abondancy maps to build hyperspectral cube

    Parameters
    ----------
    maps: array-like

    tpls: array-like
    """
    cube = np.sum(
            np.expand_dims(maps, 1) * tpls[..., np.newaxis, np.newaxis], axis=0
        )
    return cube


def lmm_cube2maps(cube: np.ndarray, tpls: np.ndarray) -> np.ndarray:
    maps = np.concatenate(
            [
                np.sum(cube * tpl[..., np.newaxis, np.newaxis], axis=0)[np.newaxis, ...]
                for tpl in tpls
            ],
            axis=0,
            )
    return maps


"""
Fourier Transform
"""
def idft(inarray: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Apply the unitary inverse Discret Fourier Transform on last two axis.

    Parameters
    ----------
    inarray: array-like
      The array to transform
    shape: tuple(int, int)
      The output shape of the last two axis.

    Notes
    -----
    Use `scipy.fft.irfftn` with `workers=-1`.
    """
    return sp.fft.irfftn(
        inarray, s=shape, axes=range(-len(shape), 0), norm="ortho", workers=-1
    )

def dft(inarray: np.ndarray) -> np.ndarray:
    """Apply the unitary Discret Fourier Transform on last two axis.

    Parameters
    ----------
    inarray: array-like
      The array to transform

    Notes
    -----
    Use `scipy.fft.rfftn` with `workers=-1`.
    """
    return sp.fft.rfftn(inarray, axes=range(-2, 0), norm="ortho", workers=-1)


"""
Interpolation
"""
def interpn_cube2local(wavel_index: np.ndarray, 
                       alpha_axis: np.ndarray, 
                       beta_axis: np.ndarray, 
                       cube: np.ndarray, 
                       local_coords: np.ndarray, 
                       local_shape: Tuple[int, int, int]) -> np.ndarray:
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
      list of 3D points of each local coordinates  in the global coordinate system
    local_shape: Tuple(int, int, int)
      3D shape of the local hyperspectral cube 
    """
    return sp.interpolate.interpn( (wavel_index, alpha_axis, beta_axis), cube, local_coords).reshape(local_shape)

def interpn_local2cube(wavel_index: np.ndarray, 
                       local_alpha_axis: np.ndarray, 
                       local_beta_axis: np.ndarray, 
                       cube: np.ndarray, 
                       global_coords: np.ndarray, 
                       global_shape: Tuple[int, int, int]) -> np.ndarray:
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
      list of 3D points of each local coordinates  in the global coordinate system
    local_shape: Tuple(int, int, int)
      3D shape of the local hyperspectral cube 
    """
    return sp.interpolate.interpn(
                                (wavel_index, local_alpha_axis, local_beta_axis),
                                cube,
                                global_coords,
                                bounds_error=False,
                                fill_value=0,
                                ).reshape(global_shape)