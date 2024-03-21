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
import scipy as sp
import udft
from aljabr import LinOp
from loguru import logger
from numpy import ndarray as array
import time
from ctypes import POINTER, c_double, c_uint32
import os
from pathlib import Path
import psutil

from . import instru
from . import cython_2D_interpolation
from surfh import cythons_files
from . import shared_dict
from . import utils

from .AsyncProcessPoolLight import APPL
from multiprocessing import Pool
from multiprocessing import Process, Queue, connection
import matplotlib.pyplot as plt

array = np.ndarray
InputShape = namedtuple("InputShape", ["wavel", "alpha", "beta"])


def dft(inarray: array) -> array:
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


def idft(inarray: array, shape: Tuple[int, int]) -> array:
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

    selected_alpha = alpha_axis[slice_alpha]
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

def wblur_t(arr: array, wpsf: array, num_threads: int) -> array:
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


class Channel(LinOp):
    """A channel with FOV, slit, spectral blurring and pce

    Attributs
    ---------
    instr: IFU
      The IFU that contains physical information.
    alpha_axis: array
      The alpha axis of the input.
    beta_axis: array
      The beta axis of the input.
    wavel_axis: array
      The wavelength axis of the input.
    srf: int
      The super resolution factor.
    pointings: `CoordList`
      The list of pointed coordinates.
    name: str
      The same name than `instr`.
    step: float
      The alpha step of alpha_axis
    wslice: slice
      The wavelength slice of input that match instr with 0.1 μm of margin.
    npix_slit: int
      The number of beta pixel inside a slit (across slit dim).
    n_alpha: int
      The number of input pixel inside a slit (along slit dim)
    local_alpha_axis, self.local_beta_axis: array
      The alpha and beta axis in local referential.
    ishape: tuple of int
      The input shape.
    oshape: tuple of int
      The output shape.
    imshape: tuple of int
      The image shape (without wavelength).
    cshape: tuplee of int
      The input cube shape (after wslice).
    local_shape: tuple of int
      The input cube shape in local referential.
    num_threads : int
      Number of threads used for parallel computation inside Channel's methods.
    """

    def __init__(
        self,
        instr: instru.IFU,
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        srf: int,
        pointings: instru.CoordList,
        shared_metadata_path: str,
        num_threads: int,
        serial: bool,
    ):
        """Forward model of a Channel

        Attributs
        ---------
        instr: IFU
          The IFU that contains physical information.
        alpha_axis: array
          The alpha axis of the input.
        beta_axis: array
          The beta axis of the input.
        wavel_axis: array
          The wavelength axis of the input.
        srf: int
          The super resolution factor.
        pointings: `CoordList`
          The list of pointed coordinates.
        num_threads : int
          Number of threads for multiproc parallelissation.
        Notes
        -----
        alpha and beta axis must have the same step and must be regular. This is
        not the case for wavel_axis that must only have incrising values.
        """

        self.old_tmp = None

        _metadata = shared_dict.attach(shared_metadata_path)
        self._metadata_path = _metadata.path
        _metadata["wavel_axis"] = wavel_axis
        _metadata["alpha_axis"] = alpha_axis
        _metadata["beta_axis"] = beta_axis


        #self.wavel_axis = wavel_axis
        #self.alpha_axis = alpha_axis
        #self.beta_axis = beta_axis

        

        if not np.allclose(alpha_axis[1] - alpha_axis[0], beta_axis[1] - beta_axis[0]):
            logger.warning(
                "α and β step for input axis must be equals. Here α={da} and β={db}",
            )

        self.pointings = pointings.pix(self.step)
        self.instr = instr.pix(self.step)

        self.num_threads = num_threads
        self.serial = serial

        self.srf = srf
        self.imshape = (len(alpha_axis), len(beta_axis))

        _metadata["_otf_sr"] = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]
        self._otf_sr = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]

        _metadata["local_alpha_axis"], _metadata["local_beta_axis"] = self.instr.fov.local_coords(
            self.step,
            alpha_margin=5 * self.step,
            beta_margin=5 * self.step,
        )

        ishape = (len(wavel_axis), self.imshape[0], self.imshape[1] // 2 + 1)
        oshape = (
            len(self.pointings),
            self.instr.n_slit,
            self.instr.n_wavel,
            ceil(self.n_alpha / self.srf),  # self.n_alpha // self.srf,
        )
        self.cshape = (
            self.wslice.stop - self.wslice.start,
            len(alpha_axis),
            len(beta_axis),
        )
        self.local_shape = (
            # self.instr.n_wavel,
            self.wslice.stop - self.wslice.start,
            len(_metadata["local_alpha_axis"]),
            len(_metadata["local_beta_axis"]),
        )

        _metadata["fw_data"] = np.zeros(oshape)
        _metadata["ad_data"] = np.zeros(ishape, dtype=np.complex128)
 
        super().__init__(ishape, oshape, self.instr.name)

        self.save_memory = False
        if not self.save_memory:
            self.precompute_wpsf()

    @property
    def step(self) -> float:
        alpha_axis = shared_dict.attach(self._metadata_path)["alpha_axis"]
        return alpha_axis[1] - alpha_axis[0]

    @property
    def beta_step(self) -> float:
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]
        return beta_axis[1] - beta_axis[0]

    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step)

    @property
    def wslice(self) -> slice:
        """The wavelength slice of input that match instr with 0.1 μm of margin."""
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        return self.instr.wslice(wavel_axis, 0.1)

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]
        return int(
            ceil(self.instr.slit_beta_width / (beta_axis[1] - beta_axis[0]))
        )

    @property
    def wavel_axis(self) -> array:
        """ """
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        return wavel_axis

    def slit_local_fov(self, slit_idx) -> instru.LocalFOV:
        """The FOV of slit `slit_idx` in local ref"""
        slit_fov = self.instr.slit_fov[slit_idx]
        return slit_fov.local + self.instr.slit_shift[slit_idx]

    def slit_slices(self, slit_idx: int) -> Tuple[slice, slice]:
        """The slices of slit `slit_idx` in local axis"""
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]
        slices = self.slit_local_fov(slit_idx).to_slices(
            local_alpha_axis, local_beta_axis
        )
        # If slice to long, remove one pixel at the beginning or the end
        if (slices[1].stop - slices[1].start) > self.npix_slit:
            if abs(
                local_beta_axis[slices[1].stop]
                - self.slit_local_fov(slit_idx).beta_end
            ) > abs(
                local_beta_axis[slices[1].start]
                - self.slit_local_fov(slit_idx).beta_start
            ):
                slices = (slices[0], slice(slices[1].start, slices[1].stop - 1))
            else:
                slices = (slices[0], slice(slices[1].start + 1, slices[1].stop))
        return slices

    def slit_shape(self, slit_idx: int) -> Tuple[int, int, int]:
        """The shape of slit `slit_idx` in local axis"""
        slices = self.slit_slices(slit_idx)
        return (
            self.wslice.stop - self.wslice.start,
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
        )

    def slit_weights(self, slit_idx: int) -> array:
        """The weights of slit `slit_idx` in local axis"""
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]
        slices = self.slit_slices(slit_idx)

        weights = fov_weight(
            self.slit_local_fov(slit_idx),
            slices,
            local_alpha_axis,
            local_beta_axis,
        )

        # If previous do not share a pixel
        if slit_idx > 0:
            if self.slit_slices(slit_idx - 1)[1].stop - 1 != slices[1].start:
                weights[:, 0] = 1

        # If next do not share a pixel
        if slit_idx < self.npix_slit - 1:
            if slices[1].stop - 1 != self.slit_slices(slit_idx + 1)[1].start:
                weights[:, -1] = 1

        return weights[np.newaxis, ...]

    def slicing(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        return gridded[:, slices[0], slices[1]] * weights

    def slicing_cube2Fente(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        return gridded[:, slices[0], slices[1]] * weights


    def slicing_t(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        out = np.zeros(self.local_shape)
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        out[:, slices[0], slices[1]] = gridded * weights
        return out

    def slicing_Fente2Cube_t(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        out = np.zeros(self.local_shape)
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        out[:, slices[0], slices[1]] = gridded* weights
        return out
        
    def gridding(self, inarray: array, pointing: instru.Coord) -> array:
        """Returns interpolation of inarray in local referential"""
        # α and β inside the FOV shifted to pointing, in the global ref.
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]
        alpha_axis = shared_dict.attach(self._metadata_path)["alpha_axis"]
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]

        alpha_coord, beta_coord = (self.instr.fov + pointing).local2global(
            local_alpha_axis, local_beta_axis
        )
        

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx),) + alpha_coord.shape
        local_coords = np.vstack(
            [
                alpha_coord.ravel(),
                beta_coord.ravel()
            ]
        ).T          

        # for i, p in enumerate(local_coords.T):
        #     if not np.logical_and(np.all(alpha_coord[i] <= p),
        #                             np.all(p <= beta_coord[i])):
        #         raise ValueError("One of the requested xi is out of bounds "
        #                             "in dimension %d" % i)
        # out_of_bounds = None

        return cython_2D_interpolation.interpn( (alpha_axis, beta_axis), 
                                              inarray, 
                                              local_coords, 
                                              len(wl_idx)).reshape(out_shape)
    

    def gridding_t(self, inarray: array, pointing: instru.Coord) -> array:
        """Returns interpolation of inarray in global referential"""
        # α and β inside the FOV shifted to pointing, in the global ref.
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]
        alpha_axis = shared_dict.attach(self._metadata_path)["alpha_axis"]
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]

        alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
            alpha_axis, beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx), len(alpha_axis), len(beta_axis))

        global_coords = np.vstack(
            [
                alpha_coord.ravel(),
                beta_coord.ravel()
            ]
        ).T
        return cython_2D_interpolation.interpn( (local_alpha_axis, local_beta_axis), 
                                              inarray, 
                                              global_coords, 
                                              len(wl_idx),
                                              bounds_error=False, 
                                              fill_value=0,).reshape(out_shape)


    def sblur(self, inarray_f: array) -> array:
        """Return spatial blurring of inarray_f in Fourier space for SR"""
        _otf_sr = shared_dict.attach(self._metadata_path)["_otf_sr"]
        return idft(
            inarray_f * _otf_sr,
            self.imshape,
        )


    def fourier_duplicate_t(self, inarray: array) -> array:
        """Return spatial blurring transpose of inarray for SR. Returns in Fourier space.
        inarray :   5. | 0. | 0. | 0. | 0. | 7. | 0. | 0. | 0. | 0. | 2. | 0. | 0. | 0. | 0. 
        _otf_sr :   1. | 1. | 1. | 1. | 1.
        
        output  :   5. | 5. | 5. | 7. | 7. | 7. | 7. | 7. | 2. | 2. | 2. | 2. | 2. | 0. | 0.
        """
        #_otf_sr = shared_dict.attach(self._metadata_path)["_otf_sr"]
        _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.imshape)[np.newaxis, ...]
        return dft(inarray) * _otf_sr.conj()

    def _wpsf(self, length: int, step: float, slit_idx: int, type: str = 'mrs') -> array:
        """Return spectral PSF"""
        # ∈ [0, β_s]
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        beta_in_slit = np.arange(0, length) * step

        if type == 'dirac':
            return self.instr.spectral_psf(
                            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
                            wavel_axis[self.wslice],
                            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
                            type='dirac',
                        )

        if self.save_memory:
            wpsf = self.instr.spectral_psf(
                            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
                            wavel_axis[self.wslice],
                            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
                            type='mrs',
                        )
        else:
            wpsf = shared_dict.attach(self._metadata_path)["wpsf"][slit_idx]
            
        return wpsf

    def wblur(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring of inarray"""
        return wblur(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx), self.num_threads if not self.serial else 1)

    def wdirac_blur(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring transpose of inarray using a dirac function.
           Only used to create generate cube from Forward data with applying Adjoint operator. """    
        return cubeToSlice(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx, 'dirac'), self.num_threads if not self.serial else 1)
   
    def wblur_t(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring transpose of inarray"""
        return wblur_t(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx), self.num_threads if not self.serial else 1)

    def wdirac_blur_t(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring transpose of inarray using a dirac function.
           Only used to create generate cube from Forward data with applying Adjoint operator. """    
        return sliceToCube_t(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx, 'dirac'), self.num_threads if not self.serial else 1)

    def forward(self, inarray_f):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = shared_dict.attach(self._metadata_path)["fw_data"]
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]* self.wblur(sliced).sum(axis=2)
               
    
    def forward_multiproc(self, inarray_f):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = shared_dict.attach(self._metadata_path)["fw_data"]
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]* self.wblur(sliced, slit_idx).sum(axis=2)

    def cubeToSlice(self, cube):
        """cube is supposed in global coordinate in Fourier space for a specific Channel and band.
        slices is an array of shape (pointing, slit, wavelength, alpha).
        Reshape input cube into slices without spatial and spectral blurring 
        done in Forward operator.
        """
        # [pointing, slit, λ', α]
        slices = np.zeros(self.oshape)
        blurred = idft(cube[self.wslice, ...], self.imshape) # Replace sblur

        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                sliced = self.slicing_cube2Fente(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                slices[p_idx, slit_idx, :, :] = self.wdirac_blur(sliced, slit_idx).sum(axis=2) #TODO Change: That
        return slices      

    def realData_cubeToSlice(self, cube):
        slices = np.zeros(self.oshape[1:]) # Remove pointing dimension
        gridded = self.gridding(cube, instru.Coord(0, 0))
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicing_cube2Fente(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
            slices[slit_idx, :, :] = sliced.sum(axis=2) # Only sum on the Beta axis
        return slices      

          

    def adjoint(self, measures):
        out = shared_dict.attach(self._metadata_path)["ad_data"]
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                tmp = np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx] * self.instr.pce[..., np.newaxis],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )

                tmp2 = self.wblur_t(tmp)

                sliced[:, : self.oshape[3] * self.srf : self.srf] = tmp2
                    
                gridded += self.slicing_t(sliced, slit_idx)
            blurred += self.gridding_t(gridded, pointing)
        
        out[self.wslice, ...] += self.fourier_duplicate_t(blurred)


    def adjoint_multiproc(self, measures):
        out = shared_dict.attach(self._metadata_path)["ad_data"]
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                tmp = np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx] * self.instr.pce[..., np.newaxis],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )

                tmp2 = self.wblur_t(tmp, slit_idx)
                sliced[:, : self.oshape[3] * self.srf : self.srf] = tmp2
                
                gridded += self.slicing_t(sliced, slit_idx)
           
            _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.local_shape[1:])[np.newaxis, ...]
            tmp = dft(gridded) * _otf_sr.conj() 
            tmp2 = idft(tmp, self.local_shape[1:])
            blurred += self.gridding_t(tmp2, pointing)
            # blurred += self.gridding_t(gridded, pointing)

        # out[self.wslice, ...] += self.fourier_duplicate_t(blurred)
        out[self.wslice, ...] += dft(blurred)


    def sliceToCube(self, measures):
        out = np.zeros(self.ishape, dtype=np.complex128)
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                tmp = np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )/sliced.shape[2] #TODO : Véririfier si diviser par la taille de beta est correct ?
                                      # Car dans le modèle direct/Adjoint c'est faux. Mais ici on fait que de la transformation
                                      # pas du modèle...

                tmp2 = self.wdirac_blur_t(tmp, slit_idx)
                sliced[:, : self.oshape[3] * self.srf : self.srf] = tmp2
            
                gridded += self.slicing_Fente2Cube_t(sliced, slit_idx)
            blurred += self.gridding_t(gridded, pointing)
        out[self.wslice, ...] = self.fourier_duplicate_t(blurred)
        return out

    def realData_sliceToCube(self, slices, cube_dim):
        out = np.zeros(cube_dim)
        blurred = np.zeros(cube_dim)
        gridded = np.zeros((cube_dim[0] , self.local_shape[1], self.local_shape[2]))
        for slit_idx in range(self.instr.n_slit):
            dlt = self.slit_slices(slit_idx)
            sliced = np.zeros((cube_dim[0], dlt[0].stop - dlt[0].start, dlt[1].stop - dlt[1].start,))
            tmp = np.repeat(
                            np.expand_dims(
                                slices[slit_idx],
                                axis=2,
                                ),
                                sliced.shape[2],
                                axis=2,
                            )/sliced.shape[2]
            tmp2 = tmp
            sliced[:, : cube_dim[0] * self.srf : self.srf] = tmp2
            # Replace slicing_Fente2Cube_t to match the right shape
            out_slice = np.zeros((cube_dim[0], self.local_shape[1], self.local_shape[2]))
            nslices = self.slit_slices(slit_idx)
            weights = self.slit_weights(slit_idx)
            out_slice[:, nslices[0], nslices[1]] = sliced* weights
            gridded += out_slice

        _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.local_shape[1:])[np.newaxis, ...]
        tmp = dft(gridded) * _otf_sr.conj() 

        # blurred += self.gridding_t(gridded, instru.Coord(0, 0))
        tmp2 = idft(tmp, self.local_shape[1:])
        
        blurred += self.gridding_t(tmp2, instru.Coord(0, 0))

        # Replace Fourier dupplicate to match the right shape
        # _otf_sr = udft.ir2fr(np.ones((1, self.srf)), cube_dim[1:])[np.newaxis, ...]
        # out = dft(blurred) * _otf_sr.conj() 
        # return idft(out, cube_dim[1:])
        return blurred

    def precompute_wpsf(self):
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]

        _metadata = shared_dict.attach(self._metadata_path)
        _metadata["wpsf"] = {}
        alpha_coord, beta_coord = (self.instr.fov).local2global(
            local_alpha_axis, local_beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange((self.wslice.stop-self.wslice.start))

        out_shape = (len(wl_idx),) + alpha_coord.shape
        gridded = np.ones(out_shape)
       
        for slit_idx in range(self.instr.n_slit):
            slices = self.slit_slices(slit_idx)
            sliced = gridded[:, slices[0], slices[1]]
            sliced = sliced[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
            
            wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
            beta_in_slit = np.arange(0, sliced.shape[2]) * self.beta_step
            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
            wpsf = self.instr.w_blur.psfs(self.instr.wavel_axis, beta_in_slit - np.mean(beta_in_slit), wavel_axis[self.wslice], arcsec2micron)
            
            _metadata["wpsf"][slit_idx] = wpsf

class Spectro(LinOp):
    def __init__(
        self,
        instrs: List[instru.IFU],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        sotf: array,
        pointings: instru.CoordList,
        verbose: bool = True,
        serial: bool = False,
    ):

        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        self.pointings = pointings
        self.verbose = verbose
        self.serial = serial

        srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step,
        )

        _shared_metadata = shared_dict.create("s_metadata_" + str(time.time()))
        self._shared_metadata = _shared_metadata
        for instr in instrs:
            _shared_metadata.addSubdict(instr.get_name_pix())

        num_threads = np.ceil(psutil.cpu_count()/len(instrs))

        self.channels = [
            Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavel_axis,
                srf,
                pointings,
                _shared_metadata[instr.get_name_pix()].path,
                num_threads,
                self.serial,
            )
            for srf, instr in zip(srfs, instrs)
        ]

        self._idx = np.cumsum([0] + [np.prod(chan.oshape) for chan in self.channels])
        self.imshape = (len(alpha_axis), len(beta_axis))
        ishape = (len(wavel_axis), len(alpha_axis), len(beta_axis))
        oshape = (self._idx[-1],)

        super().__init__(ishape, oshape, "Spectro")
        self.check_observation()

    def __del__(self):
        """ Shut down all allocated memory e.g. shared arrays and dictionnaries"""
        if self._shared_metadata is not None:
            dico = shared_dict.attach(self._shared_metadata.path)
            dico.delete() 

    @property
    def step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    def get_chan_data(self, inarray: array, chan_idx: int) -> array:
        return np.reshape(
            inarray[self._idx[chan_idx] : self._idx[chan_idx + 1]],
            self.channels[chan_idx].oshape,
        )


    def forward(self, inarray: array) -> array:
        """
            MRS forward operator.
            Apply H

            inarray : 

            output :  
        """
        out = np.zeros(self.oshape)
        if self.verbose:
            logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        for idx, chan in enumerate(self.channels):
            if self.verbose:
                logger.info(f"Channel {chan.name}")
            APPL.runJob("Forward_id:%d"%idx, chan.forward_multiproc, 
                        args=(blurred_f,), 
                        serial=self.serial)
            
        APPL.awaitJobResult("Forward*", progress=self.verbose)
        
        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            out[self._idx[idx] : self._idx[idx + 1]] = fw_data.ravel()

        return out 

    
    def adjoint(self, inarray: array) -> array:
        """
            MRS adjoint operator.
            Apply H^t 

            inarray : 

            output :  
        """
        tmp = np.zeros(
            self.ishape[:2] + (self.ishape[2] // 2 + 1,), dtype=np.complex128
        )
        for idx, chan in enumerate(self.channels):
            if self.verbose:
                logger.info(f"Channel {chan.name}")
            APPL.runJob("Adjoint_id:%d"%idx, chan.adjoint_multiproc, 
                        args=(np.reshape(inarray[self._idx[idx] : self._idx[idx + 1]], chan.oshape),), 
                        serial=self.serial)

        APPL.awaitJobResult("Adjoint*", progress=self.verbose)

        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            ad_data = self._shared_metadata[chan.name]["ad_data"]
            tmp += ad_data
        
        if self.verbose:
            logger.info(f"Spatial blurring^T : IDFT2({tmp.shape})")
        return idft(tmp * self.sotf.conj(), self.imshape)


    def cubeToSlice(self, cube):
        """
            Convert hyperspectral cube into MRS data (in slices shape).
            Similar to forward operator, without spatial and spectral blurring. 

            cube : 
                hyperspectral cube. 

            output : 
                list of MRS slices.  
        """
        slices = []
        blurred_f = dft(cube)
        for idx, chan in enumerate(self.channels):
            
            slice = chan.cubeToSlice(blurred_f)
            slices.append(slice)
        return slices
    
    def realData_cubeToSlice(self, chan_name, cube):

        return 
    


    def sliceToCube(self, slices):
        """
            Convert MRS data (in slices shape) onto a list of hyperspectral cube.
            Similar to adjoint operator, without spatial and spectral blurring. 

            Slices : 
                MRS Forward data. 

            output : 
                list of hyperspectral cube. One cube per frequency band.  
        """
        cubes = []

        for idx, chan in enumerate(self.channels):
            res = chan.sliceToCube(np.reshape(slices[self._idx[idx] : self._idx[idx + 1]], chan.oshape),)
            cube = idft(res, self.imshape)
            cubes.append(cube[chan.wslice,...])
        return cubes


    def qdcoadd(self, measures: array) -> array:
        out = np.zeros(self.ishape)
        nhit = np.zeros(self.ishape)

        def interp_l(measures, i_wl, o_wl):
            slit_idx = np.arange(measures.shape[1])
            i_coord = (i_wl, slit_idx)
            o_coord = np.vstack(  # output coordinate
                [
                    np.tile(o_wl.reshape((-1, 1)), (1, len(slit_idx))).ravel(),
                    np.tile(slit_idx.reshape((1, -1)), (len(o_wl), 1)).ravel(),
                ]
            ).T
            return sp.interpolate.interpn(
                i_coord,  # input axis
                measures,
                o_coord,
                bounds_error=False,
                fill_value=0,
            ).reshape((len(o_wl), measures.shape[1]))

        def chan_coadd(measures, chan):
            out = np.zeros(self.ishape, dtype=np.float)
            for p_idx, pointing in enumerate(self.pointings):
                gridded = np.zeros(chan.local_shape)
                for slit_idx in range(chan.instr.n_slit):
                    sliced = np.zeros(chan.slit_shape(slit_idx))
                    # λ interpolation, α repeat, and β duplication (tiling)
                    sliced[:] = np.tile(
                        np.repeat(
                            interp_l(
                                measures[p_idx, slit_idx]
                                / chan.instr.pce[..., np.newaxis],
                                chan.instr.wavel_axis,
                                self.wavel_axis[chan.wslice],
                            ),
                            repeats=chan.srf,
                            axis=1,
                        )[..., np.newaxis][:, : sliced.shape[1], :],
                        reps=(1, 1, sliced.shape[2]),
                    )
                    gridded += chan.slicing_t(sliced, slit_idx)
                out[chan.wslice, ...] += chan.gridding_t(gridded, pointing)
            return out

        for idx, chan in enumerate(self.channels):
            out += chan_coadd(
                np.reshape(measures[self._idx[idx] : self._idx[idx + 1]], chan.oshape),
                chan,
            )
            nhit += chan_coadd(
                np.reshape(
                    np.ones_like(measures[self._idx[idx] : self._idx[idx + 1]]),
                    chan.oshape,
                ),
                chan,
            )
        return out / nhit, nhit
    


    def check_observation(self):
        """ Check if channels FoV for all pointing match the observed image FoV"""

        # Get the coordinates of the observed object 
        grid = (self.alpha_axis, self.beta_axis)

        for idx, chan in enumerate(self.channels):
            # Get local alpha and beta coordinates for the channel
            local_alpha_axis = shared_dict.attach(chan._metadata_path)["local_alpha_axis"]
            local_beta_axis = shared_dict.attach(chan._metadata_path)["local_beta_axis"]
            
            for p_idx, pointing in enumerate(chan.pointings):
                out_of_bound = False
                # Get the global alpha and beta coordinates regarding the pointing for specific IFU
                alpha_coord, beta_coord = (chan.instr.fov + pointing).local2global(
                    local_alpha_axis, local_beta_axis
                )
                local_coords = np.vstack(
                            [
                                alpha_coord.ravel(),
                                beta_coord.ravel()
                            ]
                        ).T  
                
                # Check if IFU FoV anf image FoV match
                for i, p in enumerate(local_coords.T):
                    if not np.logical_and(np.all(grid[i][0] <= p),
                                        np.all(p <= grid[i][-1])):
                        out_of_bound = True

                if out_of_bound:
                    logger.debug(f"Out of bound for Chan {chan.name} - Pointing n°{p_idx}")

    def get_slits_data(self):
        """
        Return result of Forward operator in the right shape 
        [band, obs, slices, lambda, alpha]
        """
        slits = []
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            slits.append(fw_data)

        return slits

    def plot_slits_data(self, chan: int, obs : int):
        """
        Plot the data of each slit generated by the Forward operator for
        a specific band and a specific observation (one dithering). 
        """
        ifu = self.channels[chan]
        slices_data = self._shared_metadata[ifu.name]["fw_data"]

        ifu_metadata = shared_dict.attach(ifu._metadata_path)

        columns = ifu.instr.n_slit
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(8, 8), tight_layout=True)

        axs[0].imshow(slices_data[obs,0,:,:])
        #axs[0].set_yticks(ifu.instr.wavel_axis[::23])
        nx = ifu.instr.wavel_axis.shape[0]
        no_labels = 7 # how many labels to see on axis x
        step_x = int(nx / (no_labels - 1)) # step between consecutive labels
        x_positions = np.arange(0,nx,step_x)
        x_labels = np.around(ifu.instr.wavel_axis[::step_x], 2)
        axs[0].set_yticks(x_positions, x_labels)

        for i in range(1, columns):
            axs[i].imshow(slices_data[obs,i, :,:])
            axs[i].axis("off")
        plt.show()


    def plot_local_projected_data(self, chan: int, obs : int, freq : int = 0):
        """
        Plot the projection of each slit, for a specific band and observation, 
        onto a local cube.
        """
        ifu = self.channels[chan]
        out = shared_dict.attach(ifu._metadata_path)["fw_data"]

        local_cube = np.zeros((201, 17, 21)) # Shape [Lamda, alpha, beta] --> beta = n_slit, alpha = alpha_resolution

        test = out[obs, :, :, :]
        test = test[..., np.newaxis]

        for l in range(test.shape[1]):
            for s in range(test.shape[0]):
                for a in range(test.shape[2]):
                    local_cube[l, a, s] = test[s, l, a, 0]

        plt.imshow(local_cube[freq, :, :])
        plt.legend()
        plt.show()

    def close(self):
        """ Shut down all allocated memory e.g. shared arrays and dictionnaries"""
        if self._shared_metadata is not None:
            dico = shared_dict.attach(self._shared_metadata.path)
            dico.delete() 



class SpectroLMM(LinOp):
    def __init__(
        self,
        instrs: List[instru.IFU],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        sotf: array,
        pointings: instru.CoordList,
        templates: array,
        verbose: bool = True,
        serial: bool = False,
    ):

        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        self.pointings = pointings
        self.verbose = verbose
        self.serial = serial

        self.tpls = templates

        srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step,
        )

        _shared_metadata = shared_dict.create("s_metadata_" + str(time.time()))
        self._shared_metadata = _shared_metadata
        for instr in instrs:
            _shared_metadata.addSubdict(instr.get_name_pix())

        num_threads = np.ceil(psutil.cpu_count()/len(instrs))

        self.channels = [
            Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavel_axis,
                srf,
                pointings,
                _shared_metadata[instr.get_name_pix()].path,
                num_threads,
                self.serial,
            )
            for srf, instr in zip(srfs, instrs)
        ]

        self._idx = np.cumsum([0] + [np.prod(chan.oshape) for chan in self.channels])
        self.imshape = (len(alpha_axis), len(beta_axis))
        self.tpls_shape = self.tpls.shape

        # For LMM, ishape is (n_maps, n_alpha, n_beta)
        ishape = (self.tpls_shape[0], len(alpha_axis), len(beta_axis))
        oshape = (self._idx[-1],)

        super().__init__(ishape, oshape, "SpectroLMM")
        self.check_observation()

    def __del__(self):
        """ Shut down all allocated memory e.g. shared arrays and dictionnaries"""
        if self._shared_metadata is not None:
            dico = shared_dict.attach(self._shared_metadata.path)
            dico.delete() 
    

    @property
    def step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    def get_chan_data(self, inarray: array, chan_idx: int) -> array:
        return np.reshape(
            inarray[self._idx[chan_idx] : self._idx[chan_idx + 1]],
            self.channels[chan_idx].oshape,
        )

    def get_cube(self, maps):
        out = np.zeros(self.oshape)
        if self.verbose:
            logger.info(f"Cube generation")
        cube = np.sum(
            np.expand_dims(maps, 1) * self.tpls[..., np.newaxis, np.newaxis], axis=0
        )
        return cube

    def forward(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        if self.verbose:
            logger.info(f"Cube generation")
        cube = np.sum(
            np.expand_dims(inarray, 1) * self.tpls[..., np.newaxis, np.newaxis], axis=0
        )
        if self.verbose:
            logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(cube) * self.sotf
        for idx, chan in enumerate(self.channels):
            if self.verbose:
                logger.info(f"Channel {chan.name}")
            APPL.runJob("Forward_id:%d"%idx, chan.forward_multiproc, 
                        args=(blurred_f,), 
                        serial=self.serial)
            
        APPL.awaitJobResult("Forward*", progress=self.verbose)
        
        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            out[self._idx[idx] : self._idx[idx + 1]] = fw_data.ravel()

        return out 

    
    def adjoint(self, inarray: array) -> array:
        tmp = np.zeros(
            (self.wavel_axis.shape[0], self.ishape[1], self.ishape[2] // 2 + 1), dtype=np.complex128
        )
        for idx, chan in enumerate(self.channels):
            if self.verbose:
                logger.info(f"Channel {chan.name}")
            APPL.runJob("Adjoint_id:%d"%idx, chan.adjoint_multiproc, 
                        args=(np.reshape(inarray[self._idx[idx] : self._idx[idx + 1]], chan.oshape),), 
                        serial=self.serial)

        APPL.awaitJobResult("Adjoint*", progress=self.verbose)

        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            ad_data = self._shared_metadata[chan.name]["ad_data"]
            tmp += ad_data
            self._shared_metadata[chan.name]["ad_data"] = np.zeros_like(self._shared_metadata[chan.name]["ad_data"])


        if self.verbose:
            logger.info(f"Spatial blurring^T : IDFT2({tmp.shape})")
        cube = idft(tmp * self.sotf.conj(), self.imshape)
        return np.concatenate(
            [
                np.sum(cube * tpl[..., np.newaxis, np.newaxis], axis=0)[np.newaxis, ...]
                for tpl in self.tpls
            ],
            axis=0,
        )


    def sliceToCube(self, slices):
        """
            Convert MRS data (in slices shape) onto a list of hyperspectral cube.
            Similar to adjoint operator, without spatial and spectral blurring. 

            Slices : 
                MRS Forward data. 

            output : 
                list of hyperspectral cube. One cube per frequency band.  
        """
        cubes = []

        for idx, chan in enumerate(self.channels):
            res = chan.sliceToCube(np.reshape(slices[self._idx[idx] : self._idx[idx + 1]], chan.oshape),)
            cube = idft(res, self.imshape)
            cubes.append(cube[chan.wslice,...])
        return cubes


    def check_observation(self):
        """ Check if channels FoV for all pointing match the observed image FoV"""

        # Get the coordinates of the observed object 
        grid = (self.alpha_axis, self.beta_axis)

        for idx, chan in enumerate(self.channels):
            # Get local alpha and beta coordinates for the channel
            local_alpha_axis = shared_dict.attach(chan._metadata_path)["local_alpha_axis"]
            local_beta_axis = shared_dict.attach(chan._metadata_path)["local_beta_axis"]
            
            for p_idx, pointing in enumerate(chan.pointings):
                out_of_bound = False
                # Get the global alpha and beta coordinates regarding the pointing for specific IFU
                alpha_coord, beta_coord = (chan.instr.fov + pointing).local2global(
                    local_alpha_axis, local_beta_axis
                )
                local_coords = np.vstack(
                            [
                                alpha_coord.ravel(),
                                beta_coord.ravel()
                            ]
                        ).T  
                
                # Check if IFU FoV anf image FoV match
                for i, p in enumerate(local_coords.T):
                    if not np.logical_and(np.all(grid[i][0] <= p),
                                        np.all(p <= grid[i][-1])):
                        out_of_bound = True

                if out_of_bound:
                    logger.debug(f"Out of bound for Chan {chan.name} - Pointing n°{p_idx}")


    def plot_slits_data(self, chan: int, obs : int):
        """
        Plot the data of each slit generated by the Forward operator for
        a specific band and a specific observation (one dithering). 
        """
        ifu = self.channels[chan]
        slices_data = self._shared_metadata[ifu.name]["fw_data"]

        ifu_metadata = shared_dict.attach(ifu._metadata_path)

        columns = ifu.instr.n_slit
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(8, 8), tight_layout=True)

        axs[0].imshow(slices_data[obs,0,:,:])
        #axs[0].set_yticks(ifu.instr.wavel_axis[::23])
        nx = ifu.instr.wavel_axis.shape[0]
        no_labels = 7 # how many labels to see on axis x
        step_x = int(nx / (no_labels - 1)) # step between consecutive labels
        x_positions = np.arange(0,nx,step_x)
        x_labels = np.around(ifu.instr.wavel_axis[::step_x], 2)
        axs[0].set_yticks(x_positions, x_labels)

        for i in range(1, columns):
            axs[i].imshow(slices_data[obs,i, :,:])
            axs[i].axis("off")
        plt.show()


    def plot_local_projected_data(self, chan: int, obs : int, freq : int = 0):
        """
        Plot the projection of each slit, for a specific band and observation, 
        onto a local cube.
        """
        ifu = self.channels[chan]
        out = shared_dict.attach(ifu._metadata_path)["fw_data"]

        local_cube = np.zeros((201, 17, 21)) # Shape [Lamda, alpha, beta] --> beta = n_slit, alpha = alpha_resolution

        test = out[obs, :, :, :]
        test = test[..., np.newaxis]

        for l in range(test.shape[1]):
            for s in range(test.shape[0]):
                for a in range(test.shape[2]):
                    local_cube[l, a, s] = test[s, l, a, 0]

        plt.imshow(local_cube[freq, :, :])
        plt.legend()
        plt.show()


    def close(self):
        """ Shut down all allocated memory e.g. shared arrays and dictionnaries"""
        if self._shared_metadata is not None:
            dico = shared_dict.attach(self._shared_metadata.path)
            dico.delete() 
