# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import operators as op
import udft

array = np.ndarray


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

    alpha_fov: float
    beta_fov: float
    pix_size: float
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
    def slit_beta_fov(self):
        """The width of slit"""
        return self.beta_fov / self.n_slit

    def spectral_psf(self, beta, wavel_input_axis, arcsec2micron):
        """Return spectral PSF for monochromatic punctual sources

        - The number of spatial positions inside a slit can be determined given
        the slit width and the beta_axis step size.

        - Beta_axis_slit refers to the beta_axis inside a slit, it is shifted
        around it's mean value, since the detector is calibrated to have the
        maximum of spectral psf at correct value of lambda for beta=0

        """
        return self.w_blur.psfs(self.wavel_axis, beta, wavel_input_axis, arcsec2micron)


class Channel:
    """A channel with FOV, slit, spectral blurring and pce"""

    def __init__(
        self,
        channelp: ChannelParam,
        beta_step: float,
        wavel_axis: array,
        components: array,
        spsf: array,
        srf: int,
        ishape: InputShape,
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

        imshape = (ishape.alpha, ishape.beta)
        self.otf = [
            udft.ir2fr(diffracted_psf(comp, spsf, channelp.wslice, wpsf), imshape)
            * udft.ir2fr(np.ones((srf, 1)), imshape)[
                np.newaxis, ...
            ]  # * OTF for SuperResolution in alpha
            for comp in components
        ]

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        return int(np.round(self.channelp.slit_beta_fov / self.beta_step))

    def forward(self, inarray):
        # Σ_β
        for beta_idx in range(self.npix_slit):
            blurred = np.zeros(self.otf.shape[0, 0], dtype=np.complex)
            # Σ_tpl
            for idx, otf in enumerate(self.otf):
                blurred += inarray[idx][np.newaxis] * otf[beta_idx]
            blurred = np.fft.irfftn(blurred, axes=(1, 2), s=shape, norm="ortho")
            # spatial_indexing returns a cube with all slit for a specific beta
            # idx inside all slit (that should have the same number of beta)
            out += spatial_indexing(
                blurred,
                beta_idx,
                self.npix_slit,
                self.spatial_indexs[idx],
                self.channelp.n_slit,
                self.srf,
            )
        return out


def get_step(pix_size_list: List[float], pix_ratio_tol: int = 5):
    """Return the step that respect the tolerance

    that is the error is smaller than min(pix_size) / pix_ratio_tol

    >>> np.all(pix_size_list % min(pix_size / n) <= min_pix_size / ratio)

    The step is a multiple of the smallest pix_size.
    """
    num = 1
    pix_size_list = np.asarray(pix_size_list)
    min_pix_size = min(pix_size_list)
    while not np.all(
        pix_size_list % (min_pix_size / num) <= min_pix_size / pix_ratio_tol
    ):
        num += 1
    return min_pix_size / num


def get_srf(pix_size_list: List[float], step: float) -> List[int]:
    """Return the Super Resolution Factor (SRF)

    such that SRF = pix_size // step.

    Parameters
    ----------
    pix_size_list: list of float
      A list of spatial pixel size.
    step: float
      A spatial step.

    Returns
    -------
    A list of SRF int.

    """
    return [int(pix_size // step) for pix_size in pix_size_list]


Pointing = namedtuple("Pointing", ["alpha", "beta"])


class PointingList(list):
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


def diffracted_psf(component, spsf, wslice, wpsf) -> List[array]:
    """
    Parameters
    ----------
    component: array in [λ]

    spsf: array of psf in [λ, α, β]

    wslice: a Slice in λ for a channel

    wpsf : array of psf in [β_idx, λ, λ']

    shape : the spatial shape of input sky

    Returns
    =======
    A list of PSF for each

    """
    weighted_psf = spsf[wslice, ...] * component[wslice].reshape((-1, 1, 1))
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
