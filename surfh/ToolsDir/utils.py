# SURFH - SUper Resolution and Fusion for Hyperspectral images
#
# Copyright (C) 2022 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

from collections import namedtuple
from math import ceil
from typing import List, Tuple

import numpy as np
from numpy import ndarray as array
import udft
import scipy as sp

import matplotlib.pyplot as plt

def rotmatrix(degree: float) -> array:
    """Return a 2x2 rotation matrix

    Parameters
    ----------
    degree: float
       The rotation angle to apply in degree.
    """
    theta = np.radians(degree)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def gaussian_psf(wavel_axis, step, D=6.5):
    x_axis = np.linspace(-30, 30, 40).reshape((1, -1))
    y_axis = x_axis.reshape((-1, 1))
    psf = np.empty((len(wavel_axis), len(y_axis), len(y_axis)))

    for w_idx, wavel in enumerate(wavel_axis):
        fwhm_arcsec = (wavel * 1e-6 / D) * 206265  # from rad to arcsec
        sigma = fwhm_arcsec / (step * 2.354)  # sigma in pixels
        psf[w_idx] = np.exp(-(x_axis**2 + y_axis**2) / (2 * sigma**2))

    return psf / np.sum(psf, axis=(1, 2), keepdims=True)


def otf(psf, shape, components):
    otf = udft.ir2fr(
        psf[np.newaxis, ...] * components[:, :, np.newaxis, np.newaxis], shape
    )
    return otf


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




def make_mask_FoV(cube, tol=10):
    mask = np.zeros(cube.shape[1:])
    cube[cube<tol] = 0
    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            if np.any(cube[:,i,j]):
                mask[i,j] = 1
    return mask

def apply_mask_FoV(mask, cube):
    masked = mask[np.newaxis,...] * cube
    #masked[np.where(masked == 0)] = np.NaN
    return masked


def plot_maps(estimated_maps):
    nrow = 2#estimated_maps.shape[0] // 2
    ncols = estimated_maps.shape[0] // 2
    print(nrow)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncols, sharex = True, sharey = True)

    for i in range(nrow):
        for j in range(ncols):
            print(i,j)
            m = axes[i,j].imshow(estimated_maps[i*ncols+j])
            fig.colorbar(m, ax=axes[i,j])

def plot_3_cube(true_cube, y_cube, res_cube, slice=100):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex = True, sharey = True)
    m = axes[0].imshow(true_cube[slice])
    fig.colorbar(m, ax=axes[0])
    axes[0].title.set_text(f"True Cube slice n°{slice}")

    n = axes[1].imshow(y_cube[slice])
    fig.colorbar(n, ax=axes[1])
    axes[1].title.set_text(f"Data Cube slice n°{slice}")

    o = axes[2].imshow(res_cube[slice])
    fig.colorbar(o, ax=axes[2])
    axes[2].title.set_text(f"Recons Cube slice n°{slice}")

