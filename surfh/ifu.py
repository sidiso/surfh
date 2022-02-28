# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

"""IFU

IFU instrument modeling

"""

import operator as op
from dataclasses import dataclass
from math import ceil, floor
from typing import List, Tuple

import numpy as np
import udft

array = np.ndarray


def rotmatrix(degree: float) -> array:
    """Angle in degree"""
    theta = np.radians(degree)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@dataclass
class Coord:
    """A coordinate in (α, β)"""

    alpha: float
    beta: float

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

    def __sub__(self, coord: "Coord") -> "Coord":
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

    def __array__(self, dtype=None):
        """Behave as an numpy array"""
        if dtype is None:
            dtype = np.float
        return np.array([self.alpha, self.beta]).astype(dtype).reshape((2, 1))


@dataclass
class FOV:
    """A Field Of View. Angle in degree"""

    alpha_width: float
    beta_width: float
    center: Coord = Coord(0, 0)
    angle: float = 0

    def coords_self_ref(self, step: float, margin: float = 0) -> (array, array):
        """Returns cartesian coordinates inside the FOV in self referential"""

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
        """Returns cartesian coordinates inside the FOV in canonical referential"""

        alpha_coords, beta_coords = self.coords_self_ref(step, margin)

        n_alpha = alpha_coords.shape[0]
        n_beta = beta_coords.shape[1]

        coords = rotmatrix(self.angle) @ np.vstack(
            (alpha_coords.reshape((1, -1)), beta_coords.reshape((1, -1)))
        )

        return (
            coords[0].reshape((n_alpha, n_beta)) + self.center.alpha,
            coords[1].reshape((n_alpha, n_beta)) + self.center.beta,
        )

    @property
    def bbox(self):
        """The bounding box defined by the lower left un upper right point as `Coord`"""
        return (
            Coord(
                min(map(op.attrgetter("alpha"), path := self.vertices)),
                min(map(op.attrgetter("beta"), path)),
            ),
            Coord(
                max(map(op.attrgetter("alpha"), path)),
                max(map(op.attrgetter("beta"), path)),
            ),
        )

    @property
    def vertices(self):
        """The vertices as `Coord` from lower left, in counter clockwise"""
        return (self.lower_left, self.lower_right, self.upper_right, self.upper_left)

    def rotate(self, degree: float) -> None:
        """Rotation with respect to the `center`."""
        self.angle += degree

    def shift(self, coord: Coord) -> None:
        """Shift the `center`. Equivalent to `self ± Coord`."""
        self.center += coord

    @property
    def lower_left(self) -> Coord:
        """The lower left vertex"""
        return (
            Coord(-self.alpha_width / 2, -self.beta_width / 2).rotate(self.angle)
            + self.center
        )

    @property
    def lower_right(self) -> Coord:
        """The lower right vertex"""
        return (
            Coord(self.alpha_width / 2, -self.beta_width / 2).rotate(self.angle)
            + self.center
        )

    @property
    def upper_left(self) -> Coord:
        """The upper left vertex"""
        return (
            Coord(-self.alpha_width / 2, self.beta_width / 2).rotate(self.angle)
            + self.center
        )

    @property
    def upper_right(self) -> Coord:
        """The upper right vertex"""
        return (
            Coord(self.alpha_width / 2, self.beta_width / 2).rotate(self.angle)
            + self.center
        )

    def __add__(self, coord: Coord) -> "FOV":
        self.center += coord
        return self

    def __sub__(self, coord: Coord) -> "FOV":
        self.center -= coord
        return self


class SpectralBlur:
    """A spectral response"""

    def __init__(self, grating_resolution: float):
        self.grating_resolution = grating_resolution

        # the added margin serves for spectral PSF normalization. It is used in
        # private and eventually will not be taken into account at the projected
        # output
        self._n_margin = 10

    @property
    def grating_len(self):
        """The gratings length from given resolution R=λ/Δλ"""
        return 2 * 0.44245 / np.pi * self.grating_resolution

    def psfs(self, out_axis, beta, wavelength, scale: float = 1):
        """output array in [beta, wavelength, out_axis]

        scale is in arcsec2micron"""
        beta = np.asarray(beta).reshape((-1, 1, 1))
        wavelength = np.asarray(wavelength).reshape((1, -1, 1))
        out_axis = np.asarray(out_axis).reshape((1, 1, -1))

        delta_w = wavelength[1] - wavelength[0]
        # We add margin to have correct normalisation
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
        )

        # Since we are doing normalization, factor is not necessary here but we
        # keep it to have a trace of theoretical continuous normalisation
        out = (
            np.pi
            * self.grating_len
            / wavelength
            * np.sinc(
                np.pi
                * self.grating_len
                * ((out_axis - scale * beta) / w_axis_for_norm - 1)
            )
            ** 2
        )

        # Normalize in the convolution sense ("1" on the detector must comes
        # from "1" spread in the input spectrum)
        out /= np.sum(out, axis=1, keepdims=True)

        return out[:, :, self._n_margin : -self._n_margin]


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


class Slicer:
    def __init__(self, channel: float, alpha_step: float, beta_step: float):
        self.channel = channel
        self.alpha_step = alpha_step
        self.beta_step = beta_step

    def slice(self, inarray: array, pointing: Coord):
        pass


def fov2slices(fov: FOV, alpha_axis: array, beta_axis: array) -> Tuple[slice, slice]:
    alpha_step = alpha_axis[1] - alpha_axis[0]
    beta_step = beta_axis[1] - beta_axis[0]
    # If I understand well, it must be (floor, ceil) or (floor, floor) or (ceil,
    # ceil), but not round. With (floor, ceil), slit width are all
    # over-estimated, but it should be compensated by the weight.
    return (
        slice(
            int(floor(fov.start_alpha / alpha_step)),
            int(ceil(fov.end_alpha / alpha_step)),
        ),
        slice(
            int(floor(fov.start_beta / beta_step)),
            int(ceil(fov.end_beta / beta_step)),
        ),
    )


def fov_weight(
    fov: FOV, slices: Tuple[slice, slice], alpha_axis: array, beta_axis: array
) -> array:
    """Suppose the (floor, ceil) hypothesis of `fov2slices`"""
    alpha_step = alpha_axis[1] - alpha_axis[0]
    beta_step = beta_axis[1] - beta_axis[0]
    weights = np.ones(
        (slices[0].stop - slices[0].start, slices[1].stop - slices[1].start)
    )

    weights[0, :] = (
        prop := (fov.start_alpha - alpha_axis[slices[0].start]) / alpha_step
    )
    assert (
        0 <= prop <= 1
    ), "Proportion of first observed pixel in slit must be in [0, 1]"

    weights[:, 0] = (prop := (fov.start_beta - beta_axis[slices[1].start]) / beta_step)
    assert (
        0 <= prop <= 1
    ), "Proportion of first observed pixel in slit must be in [0, 1]"

    weights[-1, :] = (prop := (alpha_axis[slices[0].stop] - fov.end_alpha) / alpha_step)
    assert (
        0 <= prop <= 1
    ), "Proportion of first observed pixel in slit must be in [0, 1]"

    weights[:, -1] = (prop := (beta_axis[slices[1].stop] - fov.end_beta) / beta_step)
    assert (
        0 <= prop <= 1
    ), "Proportion of first observed pixel in slit must be in [0, 1]"

    return weights


class Channel:
    """A channel with FOV, slit, spectral blurring and pce"""

    def __init__(
        self,
        channelp: ChannelParam,
        beta_step: float,
        wavel_axis: array,
        templates: array,
        spsf: array,
        srf: int,
        ishape: "InputShape",
    ):
        self.channelp = channelp
        self.beta_step = beta_step
        self.wavel_axis = wavel_axis

        beta_in_slit = np.arange(0, self.npix_slit) * beta_step
        wpsf = self.channelp.spectral_psf(
            beta_in_slit - np.mean(beta_in_slit),
            wavel_axis,
            arcsec2micron=self.channelp.wavel_step / beta_step,
        )

        self.imshape = (ishape.alpha, ishape.beta)
        self.otf = [
            udft.ir2fr(diffracted_psf(tpl, spsf, channelp.wslice, wpsf), self.imshape)
            * udft.ir2fr(np.ones((srf, 1)), self.imshape)[
                np.newaxis, ...
            ]  # * OTF for SuperResolution in alpha
            for tpl in templates
        ]

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        return int(np.round(self.channelp.slit_beta_width / self.beta_step))

    def forward(self, inarray):
        # Σ_β
        for beta_idx in range(self.npix_slit):
            blurred = np.zeros(self.otf.shape[0, 0], dtype=np.complex)
            # Σ_tpl
            for tpl_idx, otf in enumerate(self.otf):
                blurred += inarray[tpl_idx][np.newaxis] * otf[beta_idx]
            blurred = np.fft.irfftn(blurred, axes=(1, 2), s=self.imshape, norm="ortho")
            # spatial_indexing returns a cube with all slit for a specific beta
            # idx inside all slit (that should have the same number of beta)
            out += spatial_indexing(
                blurred,
                beta_idx,
                self.npix_slit,
                self.pointed_fov[idx],
                self.channelp.n_slit,
                self.srf,
            )
        return out


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


class CoordList(list):
    """A list of Pointing with extra methods"""

    @property
    def alpha_min(self):
        """Smallest pointed α"""
        return min(map(op.attrgetter("alpha"), self))

    @property
    def beta_min(self):
        """Smallest pointed β"""
        return min(map(op.attrgetter("beta"), self))

    @property
    def alpha_max(self):
        """Largest pointed α"""
        return max(map(op.attrgetter("alpha"), self))

    @property
    def beta_max(self):
        """Largest pointed β"""
        return max(map(op.attrgetter("beta"), self))

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
    return [wblur(weighted_psf, wpsf_i) for wpsf_i in wpsf]


def wblur(arr: array, wpsf: array) -> array:
    """Apply blurring in λ axis

    Parameters
    ----------
    arr: array-like
      Input of shape [λ, α, β].
    wpsf: array-like
      Wavelength PSF of shape [λ, λ']

    Returns
    -------
    out: array-like
      A wavelength blurred array in [λ', α, β].
    """
    return np.sum(
        # in [λ, 1, α, β]
        arr.reshape((arr.shape[0], 1, arr.shape[1], arr.shape[2]))
        # wpsf_i in [λ, λ', 1, 1]
        * wpsf.reshape((wpsf.shape[0], wpsf.shape[1], 1, 1)),
        axis=0,
    )


#%% \
def fov(index) -> Tuple[float, float, float, float]:
    """Return the FOV box containing all the index

    return:
    min_alpha, max_alpha, min_beta, max_beta, min_lambda, max_lambda
    """
    min_alpha = min([idx[0].start for idx in index])
    max_alpha = max([idx[0].stop for idx in index])

    min_beta = min([idx[1].start for idx in index])
    max_beta = max([idx[1].stop for idx in index])

    return min_alpha, max_alpha, min_beta, max_beta


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
