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

"""instru

Instrument modeling

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
    """Return a 2x2 rotation matrix

    Parameters
    ----------
    degree: float
       The rotation angle to apply in degree.
    """
    theta = np.radians(degree)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def get_step(det_pix_size_list: List[float], pix_ratio_tol: int = 3):
    """Return the step that respect the tolerance

    that is the error is smaller than min(det_pix_size) / pix_ratio_tol

    >>> np.all(det_pix_size_list % min(det_pix_size / n) <= min_det_pix_size / ratio)

    The step is a multiple of the smallest det_pix_size.
    """
    num = 1
    det_pix_size_arr = np.asarray(det_pix_size_list)
    min_det_pix_size = min(det_pix_size_list)
    while not np.all(
        det_pix_size_arr % (min_det_pix_size / num) <= min_det_pix_size / pix_ratio_tol
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


@dataclass
class Coord:
    """A coordinate in (α, β)

    `Coord` can be added, and substracted (accept `+` and `-`). Inplace addition
    is also possible (`+=` and `-=`)

    `Coord` can behave as 2x1 numpy array.
    """

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
        """Return a `Coord` with rotated coordinate

        Parameter
        ---------
        degree: float
          The rotation angle in degree.

        Return
        ------
        A new `Coord` with rotation applied.
        """
        tmp = rotmatrix(degree) @ self
        return Coord(float(tmp[0]), float(tmp[1]))

    def pix(self, step: float) -> "Coord":
        """Return a new `Coord` with rounded coordinate at `step` resolution."""
        return Coord(round(self.alpha / step) * step, round(self.beta / step) * step)

    def __array__(self, dtype=None):
        """return as a 2x1 numpy array"""
        if dtype is None:
            dtype = np.float32
        return np.array([self.alpha, self.beta]).astype(dtype).reshape((2, 1))


class CoordList(list):
    """A list of `Coord` with extra methods"""

    @classmethod
    def from_array(cls, arr):
        return cls([Coord.from_array(a) for a in arr])

    @property
    def alpha_min(self) -> float:
        """Smallest pointed α"""
        return min(coord.alpha for coord in self)

    @property
    def beta_min(self) -> float:
        """Smallest pointed β"""
        return min(coord.beta for coord in self)

    @property
    def alpha_max(self) -> float:
        """Largest pointed α"""
        return max(coord.alpha for coord in self)

    @property
    def beta_max(self) -> float:
        """Largest pointed β"""
        return max(coord.beta for coord in self)

    @property
    def alpha_mean(self) -> float:
        """[max(α) + min(α)] / 2"""
        return (self.alpha_max + self.alpha_min) / 2

    @property
    def beta_mean(self) -> float:
        """[max(β) + min(β)] / 2"""
        return (self.beta_max + self.beta_min) / 2

    @property
    def alpha_box(self) -> float:
        """max(α) - min(α)"""
        return self.alpha_max - self.alpha_min

    @property
    def beta_box(self) -> float:
        """max(β) - min(β)"""
        return self.beta_max - self.beta_min

    @property
    def box(self) -> Tuple[float, float]:
        """(Δα, Δβ)"""
        return (self.alpha_box, self.beta_box)

    def fov(self, instr_list: List["IFU"], margin=5) -> "CoordList":
        """Total FOV

        Return the smallest and largest `Coord` That include all channel with
        all the pointing, with a margin.

        Parameters
        ----------
        channel_list: list of `Channel`

        margin: float
          A margin to add on each side.

        Returns
        -------
        A `CoordList` with the smallest and largest `Coord`.

        """
        alpha_min = min(instr.fov.bbox[0].alpha for instr in instr_list)
        alpha_max = max(instr.fov.bbox[1].alpha for instr in instr_list)
        beta_min = min(instr.fov.bbox[0].beta for instr in instr_list)
        beta_max = min(instr.fov.bbox[1].beta for instr in instr_list)
        return CoordList(
            [
                Coord(
                    alpha_min - self.alpha_min - margin,
                    beta_min - self.beta_min - margin,
                ),
                Coord(
                    alpha_max + self.alpha_max + margin,
                    beta_max + self.beta_max + margin,
                ),
            ]
        )

    def pix(self, step: float) -> "CoordList":
        """Return `CoordList` with rounded `Coord` to `step`."""
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
    """A Field Of View. Angle in degree

    Attributs
    ---------
    alpha_width: float
    beta_width: float
    origin: Coord
      The center of the FOV
    angle: float
      The angle in degree centered on origin

    Notes
    -----

    - "self referential:" local self ref with (0, 0) centered on `origin` and no
      angle
    - "global referential": ref in which the FOV is.

    """

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

    def global2local(self, alpha_coords, beta_coords):
        """Returns regular Cartesian global coordinates in local referential"""
        n_alpha = len(alpha_coords)
        n_beta = len(beta_coords)

        alpha_coords = alpha_coords - self.origin.alpha
        beta_coords = beta_coords - self.origin.beta

        alpha_coords = np.tile(alpha_coords.reshape((-1, 1)), [1, n_beta])
        beta_coords = np.tile(beta_coords.reshape((1, -1)), [n_alpha, 1])

        coords = rotmatrix(-self.angle) @ np.vstack(
            (alpha_coords.ravel(), beta_coords.ravel())
        )

        return (
            coords[0].reshape((n_alpha, n_beta)),
            coords[1].reshape((n_alpha, n_beta)),
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
        self.origin = self.origin + coord

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
        return round(self.origin.beta - self.beta_width / 2, 3)

    @property
    def beta_end(self):
        return round(self.origin.beta + self.beta_width / 2, 3)

    def to_slices(self, alpha_axis: array, beta_axis: array) -> Tuple[slice, slice]:
        """Return slices of axis that contains the slice

        Parameters
        ----------
        alpha_axis: array
          alpha in local referential
        beta_axis: array
          beta in local referential
        """

        alpha_step = alpha_axis[1] - alpha_axis[0]
        beta_step = beta_axis[1] - beta_axis[0]

        return (
            slice(
                np.flatnonzero(self.alpha_start < alpha_axis + alpha_step / 2)[0],
                np.flatnonzero(alpha_axis - alpha_step / 2 < self.alpha_end)[-1] + 1,
            ),
            slice(
                np.flatnonzero(self.beta_start < beta_axis + beta_step / 2)[0],
                np.flatnonzero(beta_axis - beta_step / 2 < self.beta_end)[-1] + 1,
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
    def grating_len(self) -> float:
        """The gratings length from given resolution R=λ/Δλ"""
        return 2 * 0.44245 / np.pi * self.grating_resolution

    def psfs(
        self, out_axis: array, beta: array, wavelength: array, scale: float = 1
    ) -> array:
        """Normalized discretized spetral PSF

        Parameters
        ----------
        out_axis: array
          The output axis (on detector) in μm.
        beta: array
          The beta value in arcsec.
        wavelength: array
          The input axis (on sky) in μm.
        scale: float
          The scale or conversion factor in μm / arcsec.

        Returns
        -------
        The normalized PSF (out_axis, wavelength, beta) shape.
        """
        delta_w = min(np.diff(wavelength))

        # [λ', λ, β]
        beta = np.asarray(beta).reshape((1, 1, -1))
        out_axis = np.asarray(out_axis).reshape((-1, 1, 1))
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
        ).reshape((1, -1, 1))

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
        out /= np.sum(out, axis=1, keepdims=True)

        return out[:, self._n_margin - 1 : -self._n_margin + 1, :]


@dataclass
class IFU:
    """A channel with FOV, slit, spectral blurring and pce

    Attributs
    ---------

    fov: FOV
      The field of view of instrument.
    det_pix_size: float
      The detector pixel size in arcsec.
    n_slit: int
      The number of slit inside the `fov`.
    w_blur: SpectralBlur
      The spectral blur model.
    pce: array
      The Photo Conversion Efficiency. Same shape than `wavel_axis`.
    wavel_axis: array
      The sampled wavelength on the detector. Same shape than `pce.`.
    name: str
      Optional name. Not used.
    slit_fov: list of `FOV`
      FOV of each slit.
    slit_shift: list of `Coord`
      The beta shift in local ref of each slit.
    """

    fov: FOV
    det_pix_size: float
    n_slit: int
    w_blur: SpectralBlur
    pce: array
    wavel_axis: array
    name: str = "_"

    def __post_init__(self):
        self.slit_shift = [
            Coord(0, -self.fov.beta_width / 2 + self.slit_beta_width / 2)
            + Coord(0, slit_idx * self.slit_beta_width)
            for slit_idx in range(self.n_slit)
        ]
        self.slit_fov = [
            FOV(
                alpha_width=self.fov.alpha_width,
                beta_width=self.slit_beta_width,
                origin=self.fov.origin + shift.rotate(self.fov.angle),
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

    def wslice(self, wavel_input_axis, margin=0):
        """Return the measured wavelength within a selected channel. margin in λ."""
        return slice(
            np.flatnonzero(
                wavel_input_axis <= max(self.wavel_min - margin, wavel_input_axis.min())
            )[-1],
            np.flatnonzero(
                wavel_input_axis >= min(self.wavel_max + margin, wavel_input_axis.max())
            )[0],
        )

    @property
    def slit_beta_width(self):
        """The width of slit"""
        return self.fov.beta_width / self.n_slit

    def get_name_pix(self):
        return self.name if self.name.endswith('pix') else self.name + '_pix'

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
        """Return an equivalent with rounded `origin` coordinate."""
        logger.info(f"IFU. {self.name} pixelized to {step:.2g}")
        return IFU(
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


class WavelFilter:
    def __init__(
        self, measured_wavelength: array, measured_values: array, name: str = ""
    ):
        """A wavelength filter

        Parameters
        ----------
        measured_wavelength: array-like
          The wavelength where values is acquired
        measured_values: array-like
          The measured transmittance of the filter
        """
        self.measured_wavelength = measured_wavelength
        self.measured_values = measured_values
        self.name = name

    def transmittance(self, wavelengths: array, normalized: bool = False) -> array:
        spectrum = np.interp(
            wavelengths, self.measured_wavelength, self.measured_values, left=0, right=0
        )
        if normalized:
            return spectrum / np.sum(spectrum)
        else:
            return spectrum

    def integrate_hsi(self, cube: array, wavelength: array) -> array:
        """cube is [λ, α, β] and wavelength is "λ"

        return im[α, β] = ∑_λ cube[λ, α, β] * filter[λ]"""
        return sum(
            image * weight
            for image, weight in zip(cube, self.transmittance(wavelength, True))
        )

    def integrate_spectrum(self, spectrum: array, wavelength: array) -> float:
        """Return i = ∑_λ spectrum[λ] * filter[λ]"""
        return np.sum(spectrum * self.transmittance(wavelength, True))


@dataclass
class MSImager:
    """Multi-Spectral Imager"""

    sotf: array
    fov: FOV
    wfilters: List[WavelFilter]
    det_pix_size: float


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
