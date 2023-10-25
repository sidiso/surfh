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

from . import instru
from . import cython_2D_interpolation
from . import cythons_files
from . import shared_dict

#from .CFAsyncProcessPool import CFAPP
#from .AsyncProcessPool import APP
from .AsyncProcessPoolLight import APPL
from multiprocessing import Pool
from multiprocessing import Process, Queue, connection

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


def wblur(arr: array, wpsf: array) -> array:
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

    return cythons_files.c_wblur(arr, wpsf, wpsf.shape[1], arr.shape[1], arr.shape[2], wpsf.shape[0])


def wblur_t(arr: array, wpsf: array) -> array:
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

    return cythons_files.c_wblur_t(arr, wpsf, wpsf.shape[1], arr.shape[1], arr.shape[2], wpsf.shape[0])


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

        Notes
        -----
        alpha and beta axis must have the same step and must be regular. This is
        not the case for wavel_axis that must only have incrising values.
        """


        _metadata = shared_dict.attach(shared_metadata_path)
        self._metadata_path = _metadata.path
        _metadata["wavel_axis"] = wavel_axis
        _metadata["alpha_axis"] = alpha_axis
        _metadata["beta_axis"] = beta_axis

        #self.wavel_axis = wavel_axis
        #self.alpha_axis = alpha_axis
        #self.beta_axis = beta_axis

        

        if alpha_axis[1] - alpha_axis[0] != beta_axis[1] - beta_axis[0]:
            logger.warning(
                "α and β step for input axis must be equals. Here α={da} and β={db}",
            )

        self.pointings = pointings.pix(self.step)
        self.instr = instr.pix(self.step)

        self.srf = srf
        self.imshape = (len(alpha_axis), len(beta_axis))

        _metadata["_otf_sr"] = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]
        #self._otf_sr = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]

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


    def sblur_t(self, inarray: array) -> array:
        """Return spatial blurring transpose of inarray for SR. Returns in Fourier space"""
        _otf_sr = shared_dict.attach(self._metadata_path)["_otf_sr"]
        return dft(inarray) * _otf_sr.conj()

    def _wpsf(self, length: int, step: float) -> array:
        """Return spectral PSF"""
        # ∈ [0, β_s]
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        beta_in_slit = np.arange(0, length) * step
        
        return self.instr.spectral_psf(
            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
            wavel_axis[self.wslice],
            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
        )

    def wblur(self, inarray: array) -> array:
        """Returns spectral blurring of inarray"""
        return wblur(inarray, self._wpsf(inarray.shape[2], self.beta_step))

    def wblur_t(self, inarray: array) -> array:
        """Returns spectral blurring transpose of inarray"""
        return wblur_t(inarray, self._wpsf(inarray.shape[2], self.beta_step))

    
    def forward(self, inarray_f: array):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = np.empty(self.oshape)
        blurred = self.sblur(inarray_f[self.wslice, ...])
        #logger.info(f"{self.name} : IDFT2({inarray_f.shape})")
        for p_idx, pointing in enumerate(self.pointings):
            #logger.info(
            #    f"{self.name} : gridding [{p_idx}/{len(self.pointings)}] {blurred.shape} -> {(blurred.shape[0],) + self.local_shape[1:]}"
            #)
            gridded = self.gridding(blurred, pointing)
            # Slit parallelization
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                #logger.info(f"{self.name} : slicing [{slit_idx+1}/{self.instr.n_slit}]")
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]* self.wblur(sliced).sum(axis=2)
                                
        return out


    def forward_multiproc(self, inarray_f, idx):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = shared_dict.attach(self._metadata_path)["fw_data"]
        print("Idx = ", idx)
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]* self.wblur(sliced).sum(axis=2)


    def forward_parallel(self, inarray_f: array, idx: int) -> array:
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = np.empty(self.oshape)
        print("Idx = ", idx)
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]* self.wblur(sliced).sum(axis=2)
        print("Return")   
        return
                

    def forward_cf_parallel(self, inarray_f: array, idx: int):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = np.empty(self.oshape)
        #logger.info(f"{self.name} : IDFT2({inarray_f.shape})")
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):            
            CFAPP.runJob("gridding_id:%d.%d" %(idx, p_idx), self._cf_forward_worker, args=(idx, p_idx, blurred, pointing))
                

    def _cf_forward_worker(self, idx, p_idx, blurred, pointing):
           
        out = np.empty(self.oshape[1:])
        gridded = self.gridding(blurred, pointing)
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicing(gridded, slit_idx)[
                :, : self.oshape[3] * self.srf : self.srf
            ]
            # λ blurring and Σ_β
            out[slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ] * self.wblur(sliced).sum(axis=2)

        return {"idx": idx, "p_idx": p_idx, "out" : out}

    def _forward_worker(self, idx, p_idx, blurred, pointing):
        
        out = np.empty(self.oshape[1:])
        gridded = self.gridding(blurred, pointing)
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicing(gridded, slit_idx)[
                :, : self.oshape[3] * self.srf : self.srf
            ]
            # λ blurring and Σ_β
            out[slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ] * self.wblur(sliced).sum(axis=2)

        return {"idx": idx, "p_idx": p_idx, "out" : out.ravel()}
     

    def adjoint(self, measures: array) -> array:
        out = np.zeros(self.ishape, dtype=np.complex128)
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            #logger.info(f"{self.name} : pointing [{p_idx+1}/{len(self.pointings)}]")
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                #logger.info(f"{self.name} : slicing [{slit_idx+1}/{self.instr.n_slit}]")
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                #logger.info(
                #    f"{self.name} : wblur^T {measures.shape[2:] + (sliced.shape[2],)}"
                #)
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
            #logger.info(
            #    f"{self.name} : gridding^T [{p_idx+1}/{len(self.pointings)}] {gridded.shape} -> {blurred.shape}"
            #)
            blurred += self.gridding_t(gridded, pointing)
        #logger.info(f"{self.name} : DFT2({blurred.shape})")
        out[self.wslice, ...] = self.sblur_t(blurred)
        return out

    def adjoint_multiproc(self, measures, idx, q):
        print("Idx = ", idx)
        out = np.zeros(self.ishape, dtype=np.complex128)
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            #logger.info(f"{self.name} : pointing [{p_idx+1}/{len(self.pointings)}]")
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                #logger.info(f"{self.name} : slicing [{slit_idx+1}/{self.instr.n_slit}]")
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                #logger.info(
                #    f"{self.name} : wblur^T {measures.shape[2:] + (sliced.shape[2],)}"
                #)
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
            #logger.info(
            #    f"{self.name} : gridding^T [{p_idx+1}/{len(self.pointings)}] {gridded.shape} -> {blurred.shape}"
            #)
            blurred += self.gridding_t(gridded, pointing)
        #logger.info(f"{self.name} : DFT2({blurred.shape})")
        #out[self.wslice, ...] = self.sblur_t(blurred)
        return

class Spectro_shared(LinOp):
    def __init__(
        self,
        instrs: List[instru.IFU],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        sotf: array,
        pointings: instru.CoordList,
    ):
        GD = {}

        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        GD["pointings"] = pointings
        self.pointings = pointings

        srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step,
        )

        _shared_metadata = shared_dict.create("s_metadata")
        self._shared_metadata = _shared_metadata
        for instr in instrs:
            print("Create Chan shared dict named ", instr.get_name_pix())
            _shared_metadata.addSubdict(instr.get_name_pix())

        self.channels = [
            Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavel_axis,
                srf,
                pointings,
                _shared_metadata[instr.get_name_pix()].path,
            )
            for srf, instr in zip(srfs, instrs)
        ]

        self._idx = np.cumsum([0] + [np.prod(chan.oshape) for chan in self.channels])
        self.imshape = (len(alpha_axis), len(beta_axis))
        ishape = (len(wavel_axis), len(alpha_axis), len(beta_axis))
        oshape = (self._idx[-1],)

        super().__init__(ishape, oshape, "Spectro")

    @property
    def step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    def get_chan_data(self, inarray: array, chan_idx: int) -> array:
        return np.reshape(
            inarray[self._idx[chan_idx] : self._idx[chan_idx + 1]],
            self.channels[chan_idx].oshape,
        )

    def set_general_dic(self):

        return

    def forward(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        #logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            out[self._idx[idx] : self._idx[idx + 1]] = chan.forward(blurred_f).ravel()

        return out
    
    def forward_multiproc(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        #logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        results = []
        p = []
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            p.append(Process(target=chan.forward_multiproc, args=(blurred_f, idx)))

        for proc in p:
            proc.start()

        for j in p:
            j.join()
        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            out[self._idx[idx] : self._idx[idx + 1]] = fw_data.ravel()

        return out 


    def forward_multiproc_APPL(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        #logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        results = []
        p = []
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            APPL.runJob("forward_id:%d"%idx, chan.forward_multiproc, args=(blurred_f, idx), serial=False)

        APPL.awaitJobResult()
        
        s = time.time()
        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            out[self._idx[idx] : self._idx[idx + 1]] = fw_data.ravel()
        e = time.time()
        print("Time Sort final data is ", e-s)

        return out 


    def forward_parallel(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            APP.runJob("gridding_id:%d" %(idx), chan.forward_parallel, args=(blurred_f, idx))

        APP.awaitJobResults("gridding_id:*")
        #for dicoresults in results:
        #    out[self._idx[dicoresults["idx"]] + dicoresults["p_idx"]*np.prod(dicoresults["out"].shape): self._idx[dicoresults["idx"]]+ (dicoresults["p_idx"]+1)*np.prod(dicoresults["out"].shape)] = dicoresults["out"].ravel()

        return out

    def forward_cf_parallel(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            chan.forward_cf_parallel(blurred_f, idx)

        results = CFAPP.awaitJobResults("gridding_id:*")
        for dicoresults in results:
            out[self._idx[dicoresults["idx"]] + dicoresults["p_idx"]*np.prod(dicoresults["out"].shape): self._idx[dicoresults["idx"]]+ (dicoresults["p_idx"]+1)*np.prod(dicoresults["out"].shape)] = dicoresults["out"].ravel()

        return out

    
    def adjoint(self, inarray: array) -> array:
        tmp = np.zeros(
            self.ishape[:2] + (self.ishape[2] // 2 + 1,), dtype=np.complex128
        )
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            tmp += chan.adjoint(
                np.reshape(inarray[self._idx[idx] : self._idx[idx + 1]], chan.oshape)
            )
        logger.info(f"Spatial blurring^T : IDFT2({tmp.shape})")
        return idft(tmp * self.sotf.conj(), self.imshape)


    def adjoint_multiproc(self, inarray: array) -> array:
        tmp = np.zeros(
            self.ishape[:2] + (self.ishape[2] // 2 + 1,), dtype=np.complex128
        )

        results = []
        q = Queue()
        p = []
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            p.append(Process(target=chan.adjoint_multiproc, args=(np.reshape(inarray[self._idx[idx] : self._idx[idx + 1]], chan.oshape), idx, q)))

        for proc in p:
            proc.start()

        for j in p:
            j.join()

        #logger.info(f"Spatial blurring^T : IDFT2({tmp.shape})")
        #return idft(tmp * self.sotf.conj(), self.imshape)



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
    ):
        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        self.pointings = pointings

        self.tpls = templates

        srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step,
        )

        self.channels = [
            Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavel_axis,
                srf,
                pointings,
            )
            for srf, instr in zip(srfs, instrs)
        ]

        self._idx = np.cumsum([0] + [np.prod(chan.oshape) for chan in self.channels])
        self.imshape = (len(alpha_axis), len(beta_axis))
        ishape = (len(wavel_axis), len(alpha_axis), len(beta_axis))
        oshape = (self._idx[-1],)

        super().__init__(ishape, oshape, "SpectroLMM")

    @property
    def step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    def get_chan_data(self, inarray: array, chan_idx: int) -> array:
        return np.reshape(
            inarray[self._idx[chan_idx] : self._idx[chan_idx + 1]],
            self.channels[chan_idx].oshape,
        )

    
    def forward(self, inarray: array) -> array:
        out = np.zeros(self.oshape)
        logger.info(f"Cube generation")
        cube = np.sum(
            np.expand_dims(inarray, 1) * self.tpls[..., np.newaxis, np.newaxis], axis=0
        )
        logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(cube) * self.sotf
        for idx, chan in enumerate(self.channels):
            print("nChan = ", len(self.channels))
            logger.info(f"Channel {chan.name}")
            out[self._idx[idx] : self._idx[idx + 1]] = chan.forward(blurred_f).ravel()
        return out
    
    
    def adjoint(self, inarray: array) -> array:
        tmp = np.zeros(
            self.ishape[:2] + (self.ishape[2] // 2 + 1,), dtype=np.complex128
        )
        for idx, chan in enumerate(self.channels):
            logger.info(f"Channel {chan.name}")
            tmp += chan.adjoint(
                np.reshape(inarray[self._idx[idx] : self._idx[idx + 1]], chan.oshape)
            )
        logger.info(f"Spatial blurring^T : IDFT2({tmp.shape})")
        cube = idft(tmp * self.sotf.conj(), self.imshape)
        logger.info(f"Maps summation generation")
        return np.concatenate(
            [
                np.sum(cube * tpl[..., np.newaxis, np.newaxis], axis=0)[np.newaxis, ...]
                for tpl in self.tpls
            ],
            axis=0,
        )
