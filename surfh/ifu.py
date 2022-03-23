# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

"""IFU

IFU instrument modeling

"""

from dataclasses import dataclass
from math import ceil, floor
from typing import List, Tuple

import numpy as np
import scipy.interpolate
import udft
import xarray as xr
from loguru import logger
from numpy import ndarray as array


def rotmatrix(degree: float) -> array:
    """Angle in degree"""
    theta = np.radians(degree)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@dataclass
class Coord:
    """A coordinate in (α, β)"""

    alpha: float
    beta: float

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1])

    def __add__(self, coord: "Coord") -> "Coord":
        if not isinstance(coord, Coord):
            raise ValueError("`coord` must be a `Coord`")
        return Coord(self.alpha + coord.alpha, self.beta + coord.beta)

    def __sub__(self, coord: "Coord") -> "Coord":
        if not isinstance(coord, Coord):
            raise ValueError("`coord` must be a `Coord`")
        return Coord(self.alpha - coord.alpha, self.beta - coord.beta)

    def __iadd__(self, coord: "Coord") -> "Coord":
        if not isinstance(coord, Coord):
            raise ValueError("`coord` must be a `Coord`")
        self.alpha += coord.alpha
        self.beta += coord.beta
        return self

    def __isub__(self, coord: "Coord") -> "Coord":
        if not isinstance(coord, Coord):
            raise ValueError("`coord` must be a `Coord`")
        self.alpha -= coord.alpha
        self.beta -= coord.beta
        return self

    def rotate(self, degree: float) -> "Coord":
        """Rotate the coordinate"""
        tmp = rotmatrix(degree) @ self
        self.alpha = tmp[0]
        self.beta = tmp[1]
        return self

    def pix(self, step):
        return Coord(round(self.alpha / step) * step, round(self.beta / step) * step)

    def __array__(self, dtype=None):
        """return as a numpy array"""
        if dtype is None:
            dtype = np.float
        return np.array([self.alpha, self.beta]).astype(dtype).reshape((2, 1))


class CoordList(list):
    """A list of `Coord` with extra methods"""

    @classmethod
    def from_array(cls, arr):
        return cls([Coord.from_array(a) for a in arr])

    @property
    def alpha_min(self):
        """Smallest pointed α"""
        return min(coord.alpha for coord in self)

    @property
    def beta_min(self):
        """Smallest pointed β"""
        return min(coord.beta for coord in self)

    @property
    def alpha_max(self):
        """Largest pointed α"""
        return max(coord.alpha for coord in self)

    @property
    def beta_max(self):
        """Largest pointed β"""
        return max(coord.beta for coord in self)

    @property
    def alpha_mean(self):
        """[max(α) + min(α)] / 2"""
        return (self.alpha_max + self.alpha_min) / 2

    @property
    def beta_mean(self):
        """[max(β) + min(β)] / 2"""
        return (self.beta_max + self.beta_min) / 2

    @property
    def alpha_box(self):
        """max(α) - min(α)"""
        return self.alpha_max - self.alpha_min

    @property
    def beta_box(self):
        """max(β) - min(β)"""
        return self.beta_max - self.beta_min

    @property
    def box(self):
        return (self.alpha_box, self.beta_box)

    def fov(self, channel_list: List["Channel"], margin=10):
        return (
            self.alpha_box + max(channel_list, key=lambda x: x.alpha_fov) + margin,
            self.beta_box + max(channel_list, key=lambda x: x.beta_fov) + margin,
        )

    def pix(self, step) -> "CoordList":
        """Return rounded coordinate"""
        return CoordList(c.pix(step) for c in self)

    def __array__(self, dtype=None) -> array:
        """return as a numpy array"""
        if dtype is None:
            dtype = np.float
        return (
            np.array([[c.alpha for c in self], [c.beta for c in self]])
            .astype(dtype)
            .reshape((2, -1))
        )


@dataclass
class FOV:
    """A Field Of View. Angle in degree"""

    alpha_width: float
    beta_width: float
    origin: Coord = Coord(0, 0)
    angle: float = 0

    def local_coords(
        self, step: float, alpha_margin: float = 0, beta_margin: float = 0
    ) -> Tuple[array, array]:
        """Returns regular Cartesian coordinates inside the FOV in self referential"""

        def axis(start, length, step):
            round_start = int(floor(start / step)) * step
            num = int(ceil((length + (start - round_start)) / step))
            # num = int(ceil(length / step))
            return np.arange(num + 1) * step + round_start

        alpha_axis = axis(
            -self.alpha_width / 2 - alpha_margin,
            self.alpha_width + 2 * alpha_margin,
            step,
        )
        beta_axis = axis(
            -self.beta_width / 2 - beta_margin, self.beta_width + 2 * beta_margin, step
        )

        return alpha_axis, beta_axis

    def local2global(self, alpha_coords, beta_coords):
        """Returns regular Cartesian local coordinates in global referential"""
        n_alpha = len(alpha_coords)
        n_beta = len(beta_coords)

        alpha_coords = np.tile(alpha_coords.reshape((-1, 1)), [1, n_beta])
        beta_coords = np.tile(beta_coords.reshape((1, -1)), [n_alpha, 1])

        coords = rotmatrix(self.angle) @ np.vstack(
            (alpha_coords.ravel(), beta_coords.ravel())
        )

        return (
            coords[0].reshape((n_alpha, n_beta)) + self.origin.alpha,
            coords[1].reshape((n_alpha, n_beta)) + self.origin.beta,
        )

    def coords(
        self, step: float, alpha_margin: float = 0, beta_margin: float = 0
    ) -> Tuple[array, array]:
        """Returns regular Cartesian coordinates inside the FOV in global referential"""

        alpha_coords, beta_coords = self.local_coords(step, alpha_margin, beta_margin)
        return self.local2global(alpha_coords, beta_coords)

    def rotate(self, degree: float) -> None:
        """Rotation with respect to the `origin`."""
        self.angle += degree

    def shift(self, coord: Coord) -> None:
        """Shift the `origin`. Equivalent to `self ± Coord`."""
        self.origin += coord

    @property
    def bbox(self):
        """The bounding box defined by the lower left un upper right point as `Coord`"""
        path = self.vertices
        return (
            Coord(min(p.alpha for p in path), min(p.beta for p in path)),
            Coord(max(p.alpha for p in path), max(p.beta for p in path)),
        )

    @property
    def vertices(self):
        """The vertices as `Coord` from lower left, in counter clockwise"""
        return (self.lower_left, self.lower_right, self.upper_right, self.upper_left)

    @property
    def lower_left(self) -> Coord:
        """The lower left vertex"""
        return (
            Coord(-self.alpha_width / 2, -self.beta_width / 2).rotate(self.angle)
            + self.origin
        )

    @property
    def lower_right(self) -> Coord:
        """The lower right vertex"""
        return (
            Coord(self.alpha_width / 2, -self.beta_width / 2).rotate(self.angle)
            + self.origin
        )

    @property
    def upper_left(self) -> Coord:
        """The upper left vertex"""
        return (
            Coord(-self.alpha_width / 2, self.beta_width / 2).rotate(self.angle)
            + self.origin
        )

    @property
    def upper_right(self) -> Coord:
        """The upper right vertex"""
        return (
            Coord(self.alpha_width / 2, self.beta_width / 2).rotate(self.angle)
            + self.origin
        )

    @property
    def local(self):
        """The centered FOV without angle"""
        return LocalFOV(self)

    def __add__(self, coord: Coord) -> "FOV":
        return FOV(self.alpha_width, self.beta_width, self.origin + coord, self.angle)

    def __sub__(self, coord: Coord) -> "FOV":
        return FOV(self.alpha_width, self.beta_width, self.origin - coord, self.angle)


class LocalFOV(FOV):
    def __init__(self, fov: FOV):
        super().__init__(fov.alpha_width, fov.beta_width, Coord(0, 0), angle=0)

    @property
    def alpha_start(self):
        return self.origin.alpha - self.alpha_width / 2

    @property
    def alpha_end(self):
        return self.origin.alpha + self.alpha_width / 2

    @property
    def beta_start(self):
        return self.origin.beta - self.beta_width / 2

    @property
    def beta_end(self):
        return self.origin.beta + self.beta_width / 2

    def to_slices(self, alpha_axis: array, beta_axis: array) -> Tuple[slice, slice]:
        """FOV 2 slices supposing axis are for local referential"""
        ## I don't remember why I code this below but it does not correspond to
        ## a slice of the alpha_axis parameter above. I think it's worked only
        ## with positives axis values
        # alpha_step = alpha_axis[1] - alpha_axis[0]
        # beta_step = beta_axis[1] - beta_axis[0]
        # # If I understand well, it must be (floor, ceil) or (floor, floor) or (ceil,
        # # ceil), but not round. With (floor, ceil), slit width are all
        # # over-estimated, but it should be compensated by the weight from `fov_weight`
        # return (
        #     slice(
        #         int(floor(self.alpha_start / alpha_step)),
        #         int(ceil(self.alpha_end / alpha_step)),
        #     ),
        #     slice(
        #         int(floor(self.beta_start / beta_step)),
        #         int(ceil(self.beta_end / beta_step)),
        #     ),
        # )
        return (
            slice(
                # first α below alpha_start
                np.flatnonzero(alpha_axis < self.alpha_start)[-1],
                # first α above alpha_end
                np.flatnonzero(self.alpha_end < alpha_axis)[0],
            ),
            slice(
                # first β below beta_start
                np.flatnonzero(beta_axis < self.beta_start)[-1],
                # first β above beta_end
                np.flatnonzero(self.beta_end < beta_axis)[0],
            ),
        )

    def n_alpha(self, step):
        """number of alpha in local referential"""
        return int(ceil(self.alpha_width / 2 / step)) - int(
            floor(-self.alpha_width / 2 / step)
        )

    def n_beta(self, step):
        """number of beta in local referential"""
        return int(ceil(self.beta_width / 2 / step)) - int(
            floor(-self.beta_width / 2 / step)
        )

    def __add__(self, coord: Coord) -> "LocalFOV":
        lfov = LocalFOV(self)
        lfov.origin += coord
        return lfov

    def __sub__(self, coord: Coord) -> "LocalFOV":
        lfov = LocalFOV(self)
        lfov.origin -= coord
        return lfov


class SpectralBlur:
    """A spectral response"""

    def __init__(self, grating_resolution: float):
        self.grating_resolution = grating_resolution

        # the added margin serves for spectral PSF normalization. It is used in
        # private and is removed in the projected output
        self._n_margin = 15

    @property
    def grating_len(self):
        """The gratings length from given resolution R=λ/Δλ"""
        return 2 * 0.44245 / np.pi * self.grating_resolution

    def psfs(self, out_axis, beta, wavelength, scale: float = 1):
        """output array in [beta, out_axis, wavelength]

        scale is in arcsec2micron"""
        delta_w = wavelength[1] - wavelength[0]

        # [β, λ', λ]
        beta = np.asarray(beta).reshape((-1, 1, 1))
        out_axis = np.asarray(out_axis).reshape((1, -1, 1))
        wavelength = np.asarray(wavelength)

        # Add margin to have correct normalisation
        w_axis_for_norm = np.concatenate(
            [
                np.linspace(
                    wavelength.min() - self._n_margin * delta_w,
                    wavelength.min() - delta_w,
                    self._n_margin - 1,
                ),
                wavelength,
                np.linspace(
                    wavelength.max() + delta_w,
                    wavelength.max() + self._n_margin * delta_w,
                    self._n_margin - 1,
                ),
            ],
            axis=0,
        ).reshape((1, 1, -1))

        # Since we are doing normalization, factor is not necessary here but we
        # keep it to have a trace of theoretical continuous normalisation
        out = (
            np.pi
            * self.grating_len
            / w_axis_for_norm
            * np.sinc(
                np.pi
                * self.grating_len
                * ((out_axis - scale * beta) / w_axis_for_norm - 1)
            )
            ** 2
        )

        # Normalize in the convolution sense ("1" on the detector must comes
        # from "1" spread in the input spectrum). Sum must be on the input axis.
        out /= np.sum(out, axis=2, keepdims=True)

        return out[:, :, self._n_margin - 1 : -self._n_margin + 1]


def fov_weight(
    fov: LocalFOV, slices: Tuple[slice, slice], alpha_axis: array, beta_axis: array
) -> array:
    """Suppose the (floor, ceil) hypothesis of `fov2slices`"""
    alpha_step = alpha_axis[1] - alpha_axis[0]
    beta_step = beta_axis[1] - beta_axis[0]
    slice_alpha, slice_beta = slices

    weights = np.ones(
        (slice_alpha.stop - slice_alpha.start, slice_beta.stop - slice_beta.start)
    )

    weights[0, :] *= (
        wght := (fov.alpha_start - alpha_axis[slice_alpha.start]) / alpha_step
    )
    assert (
        0 <= wght <= 1
    ), "Weight of first alpha observed pixel in slit must be in [0, 1]"

    weights[:, 0] *= (
        wght := (fov.beta_start - beta_axis[slice_beta.start]) / beta_step
    )
    assert (
        0 <= wght <= 1
    ), "Weight of first beta observed pixel in slit must be in [0, 1]"

    weights[-1, :] *= (
        wght := (alpha_axis[slice_alpha.stop] - fov.alpha_end) / alpha_step
    )
    assert (
        0 <= wght <= 1
    ), "Weight of last alpha observed pixel in slit must be in [0, 1]"

    weights[:, -1] *= (wght := (beta_axis[slice_beta.stop] - fov.beta_end) / beta_step)
    assert (
        0 <= wght <= 1
    ), "Weight of last beta observed pixel in slit must be in [0, 1]"

    return weights


@dataclass
class Instr:
    """A channel with FOV, slit, spectral blurring and pce"""

    fov: FOV
    det_pix_size: float
    n_slit: int
    w_blur: SpectralBlur
    pce: array
    wavel_axis: array
    name: str = "_"

    def __post_init__(self):
        self.slit_shift = [
            Coord(0, -self.fov.beta_width / 2)
            + Coord(0, self.slit_beta_width / 2)
            + Coord(0, slit_idx * self.slit_beta_width)
            for slit_idx in range(self.n_slit)
        ]
        self.slit_fov = [
            FOV(
                alpha_width=self.fov.alpha_width,
                beta_width=self.slit_beta_width,
                origin=self.fov.origin + shift,
                angle=self.fov.angle,
            )
            for shift in self.slit_shift
        ]

    @property
    def wavel_min(self):
        """The smallest sampled wavelength"""
        return self.wavel_axis[0]

    @property
    def wavel_max(self):
        """The largest sampled wavelength"""
        return self.wavel_axis[-1]

    @property
    def wavel_step(self):
        """The detector wavelength sampling step"""
        return self.wavel_axis[1] - self.wavel_axis[0]

    @property
    def n_wavel(self):
        """The number of detector wavelength points"""

        return len(self.wavel_axis)

    def wslice(self, wavel_input_axis):
        """Return the measured wavelength within a selected channel"""
        return slice(
            np.where(wavel_input_axis <= self.wavel_min)[0][-1],
            np.where(wavel_input_axis >= self.wavel_max)[0][0],
        )

    @property
    def slit_beta_width(self):
        """The width of slit"""
        return self.fov.beta_width / self.n_slit

    def spectral_psf(self, beta, wavel_input_axis, arcsec2micron):
        """Return spectral PSF for monochromatic punctual sources

        - The number of spatial positions inside a slit can be determined given
        the slit width and the beta_axis step size.

        - Beta_axis_slit refers to the beta_axis inside a slit, it is shifted
        around it's mean value, since the detector is calibrated to have the
        maximum of spectral psf at correct value of lambda for beta=0

        """
        return self.w_blur.psfs(self.wavel_axis, beta, wavel_input_axis, arcsec2micron)

    def pix(self, step):
        logger.info(f"Instr. {self.name} pixelized to {step:.2g}")
        return Instr(
            FOV(
                self.fov.alpha_width,
                self.fov.beta_width,
                self.fov.origin.pix(step),
                self.fov.angle,
            ),
            self.det_pix_size,
            self.n_slit,
            self.w_blur,
            self.pce,
            self.wavel_axis,
            self.name + "_pix",
        )


class Channel:
    """A channel with FOV, slit, spectral blurring and pce"""

    def __init__(
        self,
        instr: Instr,
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        templates: array,
        spsf: array,
        srf: int,
        pointings: CoordList,
        ishape: "InputShape",
    ):
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        if self.alpha_step != self.beta_step:
            logger.warning(
                "α and β step for input axis must be equal here α={da} and β={db}",
            )
        self.wavel_axis = wavel_axis

        self.pointings = pointings.pix(self.step)
        self.instr = instr.pix(self.step)

        # ∈ [0, β_s]
        beta_in_slit = np.arange(0, self.npix_slit) * self.beta_step
        wpsf = self.instr.spectral_psf(
            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
            wavel_axis,
            # suppose that physical pixel detector are square. Therefor, moving
            # on sky by Δ_β = Δ_α in β direction should move a point source by
            # one physical pixel that is Δ_λ logical pixel. This factor make the
            # conversion.
            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
        )

        self.srf = srf
        self.imshape = (ishape.alpha, ishape.beta)
        logger.info(
            f"Precompute diffrated PSF {self.instr.name} with SRF {srf} (can be optimized)",
        )
        otf_np = np.asarray(
            [
                udft.ir2fr(
                    diffracted_psf(tpl, spsf, wpsf),
                    self.imshape,
                )
                # * udft.ir2fr(np.ones((srf, 1)), self.imshape)[
                #     np.newaxis, np.newaxis, ...
                # ]  # * OTF for SuperResolution in alpha
                for tpl in templates
            ]
        )

        self.otf = xr.DataArray(otf_np, dims=("tpl", "beta", "wl_out", "nu_a", "nu_b"))

        self.local_alpha_axis, self.local_beta_axis = self.instr.fov.local_coords(
            self.step,
            alpha_margin=5 * self.step,
            beta_margin=5 * self.step,
        )

    @property
    def name(self) -> str:
        return self.instr.name

    @property
    def step(self) -> float:
        return self.alpha_step

    @property
    def alpha_step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    @property
    def beta_step(self) -> float:
        return self.beta_axis[1] - self.beta_axis[0]

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        return int(np.round(self.instr.slit_beta_width / self.beta_step))

    @property
    def n_alpha(self) -> int:
        return self.instr.fov.local.n_alpha(self.step)

    def slit(
        self,
        gridded: array,
        pointing: Coord,
        num_slit: int,
    ) -> array:
        """Return a weighted slice of gridded. num_slit start at 0."""
        slit_fov = self.instr.slit_fov[num_slit]
        local_fov = slit_fov.local + self.instr.slit_shift[num_slit]
        slices = local_fov.to_slices(self.local_alpha_axis, self.local_beta_axis)
        return gridded[:, slices[0], slices[1]] * self.slit_weight(local_fov, slices)

    def slit_weight(self, fov, slices):
        return fov_weight(
            fov,
            slices,
            self.local_alpha_axis,
            self.local_beta_axis,
        )[np.newaxis, ...]

    def gridding(self, inarray, pointing):
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
        return scipy.interpolate.interpn(
            (wl_idx, self.alpha_axis, self.beta_axis), inarray, local_coords
        ).reshape(out_shape)

    def forward(self, inarray):
        """inarray is supposed in self coordinate"""
        # First interpolate in channel self referentiel

        # [pointing, slit, λ', α]
        out = np.zeros(
            (
                len(self.pointings),
                self.instr.n_slit,
                len(self.instr.wavel_axis),
                # self.n_alpha // self.srf,
                ceil(self.n_alpha / self.srf),
            )
        )

        # Σ_β
        for beta_idx in range(self.npix_slit):
            # Σ_tpl
            blurred = sum(
                abd[np.newaxis] * otf[beta_idx] for abd, otf in zip(inarray, self.otf)
            )
            blurred = np.fft.irfftn(blurred, axes=(1, 2), s=self.imshape, norm="ortho")
            for p_idx, pointing in enumerate(self.pointings):
                gridded = self.gridding(blurred, pointing)
                for num_slit in range(self.instr.n_slit):
                    # [λ', α]
                    sliced = self.slit(gridded, pointing, num_slit)[:, :, beta_idx]
                    out[p_idx, num_slit, :, :] += np.add.reduceat(
                        sliced, range(0, sliced.shape[1], self.srf), axis=1
                    )

        return out


@dataclass
class Data:
    raw: array
    pointing: CoordList
    alpha_axis: array
    wavel_axis: array
    name: str


def get_step(det_pix_size_list: List[float], pix_ratio_tol: int = 5):
    """Return the step that respect the tolerance

    that is the error is smaller than min(det_pix_size) / pix_ratio_tol

    >>> np.all(det_pix_size_list % min(det_pix_size / n) <= min_det_pix_size / ratio)

    The step is a multiple of the smallest det_pix_size.
    """
    num = 1
    det_pix_size_list = np.asarray(det_pix_size_list)
    min_det_pix_size = min(det_pix_size_list)
    while not np.all(
        det_pix_size_list % (min_det_pix_size / num) <= min_det_pix_size / pix_ratio_tol
    ):
        num += 1
    return min_det_pix_size / num


def get_srf(det_pix_size_list: List[float], step: float) -> List[int]:
    """Return the Super Resolution Factor (SRF)

    such that SRF = det_pix_size // step.

    Parameters
    ----------
    det_pix_size_list: list of float
      A list of spatial pixel size.
    step: float
      A spatial step.

    Returns
    -------
    A list of SRF int.

    """
    return [int(det_pix_size // step) for det_pix_size in det_pix_size_list]


def diffracted_psf(template, spsf, wpsf) -> List[array]:
    """
    Parameters
    ----------
    template: array in [λ]

    spsf: array of psf in [λ, α, β]

    wpsf : array of psf in [β_idx, λ', λ]

    shape : the spatial shape of input sky

    Returns
    =======
    A list of PSF for each

    """
    weighted_psf = spsf * template.reshape((-1, 1, 1))
    return np.asarray([wblur(weighted_psf, wpsf_i) for wpsf_i in wpsf])


def wblur(arr: array, wpsf: array) -> array:
    """Apply blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ, α, β].
    wpsf: array-like
      Wavelength PSF of shape [λ', λ]

    Returns
    -------
    out: array-like
      A wavelength blurred array in [λ', α, β].
    """
    # [λ', α, β] = ∑_λ arr[λ, α, β] wpsf[λ', λ]
    # Σ_λ
    return np.sum(
        # in [1, λ, α, β]
        arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        # wpsf in [λ', λ, 1, 1]
        * wpsf.reshape((wpsf.shape[0], wpsf.shape[1], 1, 1)),
        axis=1,
    )


# #%% \
# def fov(index) -> Tuple[float, float, float, float]:
#     """Return the FOV box containing all the index

#     return:
#     min_alpha, max_alpha, min_beta, max_beta, min_lambda, max_lambda
#     """
#     min_alpha = min([idx[0].start for idx in index])
#     max_alpha = max([idx[0].stop for idx in index])

#     min_beta = min([idx[1].start for idx in index])
#     max_beta = max([idx[1].stop for idx in index])

#     return min_alpha, max_alpha, min_beta, max_beta


#%% Dither
def generate_pointings(pointing, dither):
    """pointing is tuple (alpha, beta) for central pointing.

    dither is a 2D array with dither in row and [alpha, beta] in
    column

    return a list of tuple that contains all absolute pointing"""
    return [(pointing[0] + dith[0], pointing[1] + dith[1]) for dith in dither]


### Local Variables:
### ispell-local-dictionary: "english"
### End:
