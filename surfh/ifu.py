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

    def coords_self_ref(self, step: float, margin: float = 0) -> (array, array):
        """Returns uniform Cartesian coordinates inside the FOV in self referential"""

        def axis(start, length, step):
            round_start = int(floor(start / step)) * step
            num = int(ceil((length + (start - round_start)) / step))
            # num = int(ceil(length / step))
            return np.arange(num + 1) * step + round_start

        alpha_axis = np.reshape(
            axis(-self.alpha_width / 2 - margin, self.alpha_width + 2 * margin, step),
            (-1, 1),
        )
        beta_axis = np.reshape(
            axis(-self.beta_width / 2 - margin, self.beta_width + 2 * margin, step),
            (1, -1),
        )
        n_alpha = alpha_axis.shape[0]
        n_beta = beta_axis.shape[1]

        return (np.tile(alpha_axis, [1, n_beta]), np.tile(beta_axis, [n_alpha, 1]))

    def coords(self, step: float, margin: float = 0) -> (array, array):
        """Returns uniform Cartesian coordinates inside the FOV in global referential"""

        alpha_coords, beta_coords = self.coords_self_ref(step, margin)

        n_alpha = alpha_coords.shape[0]
        n_beta = beta_coords.shape[1]

        coords = rotmatrix(self.angle) @ np.vstack(
            (alpha_coords.reshape((1, -1)), beta_coords.reshape((1, -1)))
        )

        return (
            coords[0].reshape((n_alpha, n_beta)) + self.origin.alpha,
            coords[1].reshape((n_alpha, n_beta)) + self.origin.beta,
        )

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
        """Return the same FOV but without angle"""
        return LocalFOV(self)

    def __add__(self, coord: Coord) -> "FOV":
        self.origin += coord
        return self

    def __sub__(self, coord: Coord) -> "FOV":
        self.origin -= coord
        return self


class LocalFOV(FOV):
    def __init__(self, fov: FOV):
        super().__init__(fov.alpha_width, fov.beta_width, fov.origin, angle=0)

    @property
    def alpha_start(self):
        return self.origin.alpha - self.alpha_width / 2

    @property
    def alpha_end(self):
        return self.origin.alpha + self.alpha_width / 2

    @property
    def beta_start(self):
        return self.origin.beta - self.alpha_width / 2

    @property
    def beta_end(self):
        return self.origin.beta + self.alpha_width / 2

    def to_slices(self, alpha_axis: array, beta_axis: array) -> Tuple[slice, slice]:
        """FOV 2 slices supposing axis are for local referential"""
        alpha_step = alpha_axis[1] - alpha_axis[0]
        beta_step = beta_axis[1] - beta_axis[0]
        # If I understand well, it must be (floor, ceil) or (floor, floor) or (ceil,
        # ceil), but not round. With (floor, ceil), slit width are all
        # over-estimated, but it should be compensated by the weight from `fov_weight`
        return (
            slice(
                int(floor(self.alpha_start / alpha_step)),
                int(ceil(self.alpha_end / alpha_step)),
            ),
            slice(
                int(floor(self.beta_start / beta_step)),
                int(ceil(self.beta_end / beta_step)),
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


class SpectralBlur:
    """A spectral response"""

    def __init__(self, grating_resolution: float):
        self.grating_resolution = grating_resolution

        # the added margin serves for spectral PSF normalization. It is used in
        # private and is removed in the projected output
        self._n_margin = 10

    @property
    def grating_len(self):
        """The gratings length from given resolution R=λ/Δλ"""
        return 2 * 0.44245 / np.pi * self.grating_resolution

    def psfs(self, out_axis, beta, wavelength, scale: float = 1):
        """output array in [beta, wavelength, out_axis]

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

        return out[:, :, self._n_margin : -self._n_margin]


def fov_weight(
    fov: FOV, slices: Tuple[slice, slice], alpha_axis: array, beta_axis: array
) -> array:
    """Suppose the (floor, ceil) hypothesis of `fov2slices`"""
    alpha_step = alpha_axis[1] - alpha_axis[0]
    beta_step = beta_axis[1] - beta_axis[0]
    slice_alpha, slice_beta = slices

    weights = np.ones(
        (slice_alpha.stop - slice_alpha.start, slice_beta.stop - slice_beta.start)
    )

    weights[0, :] = (
        wght := (fov.start_alpha - alpha_axis[slice_alpha.start]) / alpha_step
    )
    assert (
        0 <= wght <= 1
    ), "Weight of first alpha observed pixel in slit must be in [0, 1]"

    weights[:, 0] = (wght := (fov.start_beta - beta_axis[slice_beta.start]) / beta_step)
    assert (
        0 <= wght <= 1
    ), "Weight of first beta observed pixel in slit must be in [0, 1]"

    weights[-1, :] = (
        wght := (alpha_axis[slice_alpha.stop] - fov.end_alpha) / alpha_step
    )
    assert (
        0 <= wght <= 1
    ), "Weight of last alpha observed pixel in slit must be in [0, 1]"

    weights[:, -1] = (wght := (beta_axis[slice_beta.stop] - fov.end_beta) / beta_step)
    assert (
        0 <= wght <= 1
    ), "Weight of last beta observed pixel in slit must be in [0, 1]"

    return weights


@dataclass
class ChannelParam:
    """A channel with FOV, slit, spectral blurring and pce"""

    fov: FOV
    det_pix_size: float
    n_slit: int
    w_blur: SpectralBlur
    pce: array
    wavel_axis: array
    name: str = "_"

    def __post_init__(self):
        self.slit_fov = list(
            FOV(
                alpha_width=self.fov.alpha_width,
                beta_width=self.slit_beta_width,
                origin=self.fov.origin
                - Coord(0, self.fov.beta_width / 2)
                + Coord(0, self.slit_beta_width / 2)
                + Coord(0, slit_idx * self.slit_beta_width),
                angle=self.fov.angle,
            )
            for slit_idx in range(self.n_slit)
        )

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

    # def slit_fov(self, num_slit):
    #     """num_slit must start at 0"""
    #     return (
    #         FOV(self.fov.alpha_width, self.slit_beta_width)
    #         + num_slit * self.slit_beta_width
    #         + self.slit_beta_width / 2
    #     )

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
        logger.info(f"Channel {self.name} pixelized to {step:.2g}")
        return ChannelParam(
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
        channelp: ChannelParam,
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

        self.channelp = channelp.pix(self.step)

        # ∈ [0, β_s]
        beta_in_slit = np.arange(0, self.npix_slit) * self.beta_step
        wpsf = self.channelp.spectral_psf(
            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
            wavel_axis,
            # suppose that physical pixel detector are square. Therefor, moving
            # by Δ_β = Δ_α in β direction should move a point source by one
            # physical pixel that is Δ_λ logical pixel. This factor make the
            # conversion.
            arcsec2micron=self.channelp.wavel_step / self.beta_step,
        )

        self.srf = srf
        self.imshape = (ishape.alpha, ishape.beta)
        logger.info(
            f"Precompute diffrated PSF {self.channelp.name} with SRF {srf}",
        )
        otf_np = np.asarray(
            [
                udft.ir2fr(
                    diffracted_psf(tpl, spsf, channelp.wslice(wavel_axis), wpsf),
                    self.imshape,
                )
                * udft.ir2fr(np.ones((srf, 1)), self.imshape)[
                    np.newaxis, np.newaxis, ...
                ]  # * OTF for SuperResolution in alpha
                for tpl in templates
            ]
        )

        self.otf = xr.DataArray(otf_np, dims=("tpl", "beta", "lout", "nu_a", "nu_b"))

        self.local_alpha_axis, self.local_beta_axis = self.channelp.fov.coords_self_ref(
            self.step
        )
        self.pointings = pointings.pix(self.step)

    @property
    def name(self) -> str:
        return self.channelp.name

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
        return int(np.round(self.channelp.slit_beta_width / self.beta_step))

    @property
    def n_alpha(self) -> int:
        return self.channelp.fov.local.n_alpha(self.step)

    def slit(
        self,
        gridded: array,
        pointing: Coord,
        num_slit: int,
    ) -> array:
        """Return a weighted slice of gridded. num_slit start at 0."""
        slit_fov = self.channelp.slit_fov[num_slit] + pointing
        slices = slit_fov.local.to_slices(
            self.local_alpha_axis[:, 0], self.local_beta_axis[0, :]
        )
        return gridded[:, slices[0], slices[1]] * self.slit_weight(num_slit, pointing)

    def slit_weight(self, num_slit, pointing):
        fov = self.channelp.slit_fov[num_slit] + pointing
        return fov_weight(
            fov,
            fov.local.to_slices(
                self.local_alpha_axis[:, 0], self.local_beta_axis[0, :]
            ),
            self.local_alpha_axis,
            self.local_beta_axis,
        )

    def gridding(self, inarray):
        alpha_coord, beta_coord = self.channelp.fov.coords(self.step)
        l_coord = np.arange(inarray.shape[0])

        oshape = (len(l_coord),) + alpha_coord.shape

        new_lab = np.vstack(
            [
                np.repeat(
                    np.repeat(l_coord.reshape((-1, 1, 1)), oshape[1], axis=1),
                    oshape[2],
                    axis=2,
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], oshape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], oshape[0], axis=0).ravel(),
            ]
        ).T

        return scipy.interpolate.interpn(
            (l_coord, self.alpha_axis, self.beta_axis), inarray, new_lab
        ).reshape(oshape)

    def forward(self, inarray):
        """inarray is supposed in self coordinate"""
        # First interpolate in channel self referentiel

        out = np.zeros(
            (
                len(self.pointings),
                self.channelp.n_slit,
                len(self.wavel_axis),
                self.n_alpha // self.srf,
            )
        )

        # Σ_β
        for beta_idx in range(self.npix_slit):
            # Σ_tpl
            blurred = sum(
                abd[np.newaxis] * otf[beta_idx] for abd, otf in zip(inarray, self.otf)
            )
            blurred = np.fft.irfftn(blurred, axes=(1, 2), s=self.imshape, norm="ortho")
            gridded = self.gridding(blurred)
            for p_idx, pointing in enumerate(self.pointings):
                for num_slit in range(self.channelp.n_slit):
                    out[p_idx, num_slit, :, :] += self.slit(
                        gridded, pointing, num_slit
                    )[:, :: self.srf, beta_idx]

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


def diffracted_psf(template, spsf, wslice, wpsf) -> List[array]:
    """
    Parameters
    ----------
    template: array in [λ]

    spsf: array of psf in [λ, α, β]

    wslice: a Slice in λ for a channel

    wpsf : array of psf in [β_idx, λ, λ']

    shape : the spatial shape of input sky

    Returns
    =======
    A list of PSF for each

    """
    weighted_psf = spsf[wslice, ...] * template[wslice].reshape((-1, 1, 1))
    return np.asarray([wblur(weighted_psf, wpsf_i[:, wslice]) for wpsf_i in wpsf])


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
