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
from typing import List, Tuple
import numpy as np
from numpy import ndarray as array
from numba import njit, prange

from surfh.Models import instru
from surfh.ToolsDir import cythons_files
from surfh.Others.AsyncProcessPoolLight import APPL



InputShape = namedtuple("InputShape", ["wavel", "alpha", "beta"])

def fov_weight(
    fov: instru.LocalFOV,
    slices: Tuple[slice, slice],
    alpha_axis: array,
    beta_axis: array,
) -> array:
    """The weight windows of the FOV given slices of axis

    Notes
    -----
    Suppose the (floor, ceil) hypothesis of `LocalFOV.to_slices`.
    """
    alpha_step = alpha_axis[1] - alpha_axis[0]
    beta_step = beta_axis[1] - beta_axis[0]
    slice_alpha, slice_beta = slices

    selected_beta = beta_axis[slice_beta]

    weights = np.ones(
        (slice_alpha.stop - slice_alpha.start, slice_beta.stop - slice_beta.start)
    )

    # Weight for first α for all β
    # weights[0, :] *= (
    #     wght := abs(selected_alpha[0] - alpha_step / 2 - fov.alpha_start) / alpha_step
    # )
    # assert (
    #     0 <= wght <= 1
    # ), f"Weight of first alpha observed pixel in slit must be in [0, 1] ({wght:.2f})"

    if selected_beta[0] - beta_step / 2 < fov.beta_start:
        weights[:, 0] = (
            wght := 1
            - abs(selected_beta[0] - beta_step / 2 - fov.beta_start) / beta_step
        )
        assert (
            0 <= wght <= 1
        ), f"Weight of first beta observed pixel in slit must be in [0, 1] ({wght:.2f})"

    # weights[-1, :] *= (
    #     wght := abs(selected_alpha[-1] + alpha_step / 2 - fov.alpha_end) / alpha_step
    # )
    # assert (
    #     0 <= wght <= 1
    # ), f"Weight of last alpha observed pixel in slit must be in [0, 1] ({wght:.2f})"

    if selected_beta[-1] + beta_step / 2 > fov.beta_end:
        weights[:, -1] = (
            wght := 1
            - abs(selected_beta[-1] + beta_step / 2 - fov.beta_end) / beta_step
        )
        assert (
            0 <= wght <= 1
        ), f"Weight of last beta observed pixel in slit must be in [0, 1] ({wght:.2f})"

    return weights

def wblur(arr: array, wpsf: array, num_threads: int) -> array:
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
    result_array = cythons_files.c_wblur(np.ascontiguousarray(arr), 
                                         np.ascontiguousarray(wpsf), 
                                         wpsf.shape[1], arr.shape[1], 
                                         arr.shape[2], wpsf.shape[0],
                                         num_threads)
    return result_array

def cubeToSlice(arr: array, dirac: array, num_threads: int) -> array:
    """Apply transpose of blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ', α, β].
    Returns
    -------
    out: array-like
      A wavelength blurred array in [λ, α, β].
    """
    # [λ, α, β] = ∑_λ' arr[λ', α, β]
    # Σ_λ'
    result_array = cythons_files.c_cubeToSlice(arr, dirac, dirac.shape[1],
                                         arr.shape[1], arr.shape[2], 
                                         dirac.shape[0], num_threads)
    return result_array

# def wblur_t(arr: array, wpsf: array, num_threads: int) -> array:
#     """Apply transpose of blurring in λ axis

#     Parameters
#     ----------
#     arr: array-like
#       Input of shape [λ', α, β].
#     wpsf: array-like
#       Wavelength PSF of shape [λ', λ, β]

#     Returns
#     -------
#     out: array-like
#       A wavelength blurred array in [λ, α, β].
#     """
#     # [λ, α, β] = ∑_λ' arr[λ', α, β] wpsf[λ', λ]
#     # Σ_λ'
#     result_array = cythons_files.c_wblur_t(arr, wpsf, wpsf.shape[1], 
#                                            arr.shape[1], arr.shape[2], 
#                                            wpsf.shape[0], num_threads)
#     return result_array



def sliceToCube_t(arr: array, dirac: array, num_threads: int) -> array:
    """Apply transpose of blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ', α, β].
    Returns
    -------
    out: array-like
      A wavelength blurred array in [λ, α, β].
    """
    # [λ, α, β] = ∑_λ' arr[λ', α, β]
    # Σ_λ'
    result_array = cythons_files.c_sliceToCube_t(arr, dirac, dirac.shape[1], 
                                           arr.shape[1], arr.shape[2], 
                                           dirac.shape[0], num_threads)
    return result_array


def diffracted_psf(template, spsf, wpsf) -> List[array]:
    """
    Parameters
    ----------
    template: array in [λ]

    spsf: array of psf in [λ, α, β]

    wpsf : array of psf in [λ', λ, β]

    shape : the spatial shape of input sky

    Returns
    =======
    A list of PSF for each

    """
    weighted_psf = spsf * template.reshape((-1, 1, 1))
    return wblur(weighted_psf, wpsf)


def linearMixingModel_maps2cube(maps, NLambda, ishape, tpls):
    cube = cythons_files.c_fast_LMM_maps2cube(NLambda,ishape[0], 
                                              ishape[1], ishape[2],
                                              tpls.astype(np.float32), maps.astype(np.float32))
    return np.array(cube)

@njit(parallel=True)
def linearMixingModel_cube2maps(cube, NLambda, ishape, tpls):
    maps = np.zeros(ishape, dtype=tpls.dtype)
    tmp = 0
    for m in range(ishape[0]):
        for i in prange(ishape[1]):
            for j in range(ishape[2]):
                for lam in range(NLambda):
                    tmp += cube[lam, i, j]*tpls[m, lam]
                maps[m,i,j] = tmp
                tmp = 0
    return maps

@njit(parallel=True)
def wblur_subSampling(arr, wpsf):
    """
    Fonction équivalente en Numba pour flou et sous-échantillonnage.
    Args:
        arr: Tableau numpy de dimensions [λ, α, β]
        wpsf: Tableau numpy de dimensions [λ', λ, β]
    Returns:
        Tableau de dimensions [λ', α] représentant la somme pondérée.
    """
    # Dimensions des tableaux
    lambda_prime, lambda_dim, beta = wpsf.shape
    _, alpha, _ = arr.shape
    
    # Résultat accumulé
    result = np.zeros((lambda_prime, alpha), dtype=arr.dtype)
    
    # Calcul
    for l_p in range(lambda_prime):  # Parcourt λ'
        for a in range(alpha):       # Parcourt α
            for b in range(beta):    # Parcourt β
                for l in range(lambda_dim):  # Parcourt λ
                    result[l_p, a] += arr[l, a, b] * wpsf[l_p, l, b]
    
    return result


@njit(parallel=True)
def wblur_t(arr, wpsf):
    """
    Fonction équivalente en Numba pour le flou avec transposition.
    Args:
        arr: Tableau numpy de dimensions [λ', α, β]
        wpsf: Tableau numpy de dimensions [λ', λ, β]
    Returns:
        Tableau de dimensions [λ, α, β] représentant la somme pondérée.
    """
    # Dimensions des tableaux
    lambda_prime, alpha, beta = arr.shape
    _, lambda_dim, _ = wpsf.shape
    
    # Résultat accumulé
    result = np.zeros((lambda_dim, alpha, beta), dtype=arr.dtype)
    
    # Calcul
    for l in range(lambda_dim):  # Parcourt λ
        for a in range(alpha):   # Parcourt α
            for b in range(beta): # Parcourt β
                for l_p in range(lambda_prime):  # Parcourt λ'
                    result[l, a, b] += arr[l_p, a, b] * wpsf[l_p, l, b]
    
    return result
