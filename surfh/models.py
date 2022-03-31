# Copyright (C) 2022 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

from collections import namedtuple
from math import ceil
from typing import List, Tuple

import numpy as np
import scipy as sp
import udft
from aljabr import LinOp
from loguru import logger
from numpy import ndarray as array

from . import ifu

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
    fov: ifu.LocalFOV, slices: Tuple[slice, slice], alpha_axis: array, beta_axis: array
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
    # [λ', α, β] = ∑_λ arr[λ, α, β] wpsf[λ', λ]
    # Σ_λ
    return np.sum(
        # in [1, λ, α, β]
        np.expand_dims(arr, axis=0)
        # wpsf in [λ', λ, 1, β]
        * np.expand_dims(wpsf, axis=2),
        axis=1,
    )


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
    return np.sum(
        # in [λ', 1, α, β]
        np.expand_dims(arr, axis=1)
        # wpsf in [λ', λ, 1, β]
        * np.expand_dims(wpsf, axis=2),
        axis=0,
    )


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
    instr: Instr
      The Instr that contains physical information.
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
        instr: ifu.Instr,
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        srf: int,
        pointings: ifu.CoordList,
    ):
        """Forward model of a Channel

        Attributs
        ---------
        instr: Instr
          The Instr that contains physical information.
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
        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        if alpha_axis[1] - alpha_axis[0] != beta_axis[1] - beta_axis[0]:
            logger.warning(
                "α and β step for input axis must be equals. Here α={da} and β={db}",
            )

        self.pointings = pointings.pix(self.step)
        self.instr = instr.pix(self.step)

        self.srf = srf
        self.imshape = (len(alpha_axis), len(beta_axis))
        self._otf_sr = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]

        self.local_alpha_axis, self.local_beta_axis = self.instr.fov.local_coords(
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
            len(self.local_alpha_axis),
            len(self.local_beta_axis),
        )
        super().__init__(ishape, oshape, self.instr.name)

    @property
    def step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    @property
    def beta_step(self) -> float:
        return self.beta_axis[1] - self.beta_axis[0]

    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step)

    @property
    def wslice(self) -> slice:
        """The wavelength slice of input that match instr with 0.1 μm of margin."""
        return self.instr.wslice(self.wavel_axis, 0.1)

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        return int(
            ceil(self.instr.slit_beta_width / (self.beta_axis[1] - self.beta_axis[0]))
        )

    def slit_local_fov(self, slit_idx) -> ifu.LocalFOV:
        """The FOV of slit `slit_idx` in local ref"""
        slit_fov = self.instr.slit_fov[slit_idx]
        return slit_fov.local + self.instr.slit_shift[slit_idx]

    def slit_slices(self, slit_idx: int) -> Tuple[slice, slice]:
        """The slices of slit `slit_idx` in local axis"""
        slices = self.slit_local_fov(slit_idx).to_slices(
            self.local_alpha_axis, self.local_beta_axis
        )
        # If slice to long, remove one pixel at the beginning or the end
        if (slices[1].stop - slices[1].start) > self.npix_slit:
            if abs(
                self.local_beta_axis[slices[1].stop]
                - self.slit_local_fov(slit_idx).beta_end
            ) > abs(
                self.local_beta_axis[slices[1].start]
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
        slices = self.slit_slices(slit_idx)

        weights = fov_weight(
            self.slit_local_fov(slit_idx),
            slices,
            self.local_alpha_axis,
            self.local_beta_axis,
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

    def gridding(self, inarray: array, pointing: ifu.Coord) -> array:
        """Returns interpolation of inarray in local referential"""
        # α and β inside the FOV shifted to pointing, in the global ref.
        alpha_coord, beta_coord = (self.instr.fov + pointing).local2global(
            self.local_alpha_axis, self.local_beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx),) + alpha_coord.shape

        local_coords = np.vstack(
            [
                np.repeat(
                    np.repeat(wl_idx.reshape((-1, 1, 1)), out_shape[1], axis=1),
                    out_shape[2],
                    axis=2,
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], out_shape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], out_shape[0], axis=0).ravel(),
            ]
        ).T

        # This output can be processed in local ref.
        return sp.interpolate.interpn(
            (wl_idx, self.alpha_axis, self.beta_axis), inarray, local_coords
        ).reshape(out_shape)

    def gridding_t(self, inarray: array, pointing: ifu.Coord) -> array:
        """Returns interpolation of inarray in global referential"""
        # α and β inside the FOV shifted to pointing, in the global ref.
        alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
            self.alpha_axis, self.beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx), len(self.alpha_axis), len(self.beta_axis))

        global_coords = np.vstack(
            [
                np.tile(
                    wl_idx.reshape((-1, 1, 1)), (1, out_shape[1], out_shape[2])
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], out_shape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], out_shape[0], axis=0).ravel(),
            ]
        ).T

        # This output can be processed in local ref.
        return sp.interpolate.interpn(
            (wl_idx, self.local_alpha_axis, self.local_beta_axis),
            inarray,
            global_coords,
            bounds_error=False,
            fill_value=0,
        ).reshape(out_shape)

    def sblur(self, inarray_f: array) -> array:
        """Return spatial blurring of inarray_f in Fourier space for SR"""
        return idft(
            inarray_f * self._otf_sr,
            self.imshape,
        )

    def sblur_t(self, inarray: array) -> array:
        """Return spatial blurring transpose of inarray for SR. Returns in Fourier space"""
        return dft(inarray) * self._otf_sr.conj()

    def _wpsf(self, length: int, step: float) -> array:
        """Return spectral PSF"""
        # ∈ [0, β_s]
        beta_in_slit = np.arange(0, length) * step
        return self.instr.spectral_psf(
            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
            self.wavel_axis[self.wslice],
            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
        )

    def wblur(self, inarray: array) -> array:
        """Returns spectral blurring of inarray"""
        return wblur(inarray, self._wpsf(inarray.shape[2], self.beta_step))

    def wblur_t(self, inarray: array) -> array:
        """Returns spectral blurring transpose of inarray"""
        return wblur_t(inarray, self._wpsf(inarray.shape[2], self.beta_step))

    def forward(self, inarray_f: array) -> array:
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Outoutp is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = np.empty(self.oshape)
        logger.info(f"{self.name} : IDFT2({inarray_f.shape})")
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            logger.info(
                f"{self.name} : gridding [{p_idx}/{len(self.pointings)}] {blurred.shape} -> {(blurred.shape[0],) + self.local_shape[1:]}"
            )
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                logger.info(f"{self.name} : slicing [{slit_idx+1}/{self.instr.n_slit}]")
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                # λ blurring and Σ_β
                logger.info(f"{self.name} : wblur {sliced.shape} → {out.shape[2:]}")
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ] * self.wblur(sliced).sum(axis=2)
        return out

    def adjoint(self, measures: array) -> array:
        out = np.zeros(self.ishape, dtype=np.complex128)
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            logger.info(f"{self.name} : pointing [{p_idx+1}/{len(self.pointings)}]")
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                logger.info(f"{self.name} : slicing [{slit_idx+1}/{self.instr.n_slit}]")
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                logger.info(
                    f"{self.name} : wblur^T {measures.shape[2:] + (sliced.shape[2],)}"
                )
                sliced[:, : self.oshape[3] * self.srf : self.srf] = self.wblur_t(
                    np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx] * self.instr.pce[..., np.newaxis],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )
                )
                gridded += self.slicing_t(sliced, slit_idx)
            logger.info(
                f"{self.name} : gridding^T [{p_idx+1}/{len(self.pointings)}] {gridded.shape} -> {blurred.shape}"
            )
            blurred += self.gridding_t(gridded, pointing)
        logger.info(f"{self.name} : DFT2({blurred.shape})")
        out[self.wslice, ...] = self.sblur_t(blurred)
        return out


class Spectro(LinOp):
    def __init__(
        self,
        instrs: List[ifu.Instr],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        sotf: array,
        pointings: ifu.CoordList,
    ):
        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        self.pointings = pointings

        srfs = ifu.get_srf(
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

        super().__init__(ishape, oshape, "Spectro")

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
        logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        for idx, chan in enumerate(self.channels):
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
        return idft(tmp * self.sotf.conj(), self.imshape)


class SpectroLMM(LinOp):
    def __init__(
        self,
        instrs: List[ifu.Instr],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        sotf: array,
        pointings: ifu.CoordList,
        templates: array,
    ):
        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        self.pointings = pointings

        self.tpls = templates

        srfs = ifu.get_srf(
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
