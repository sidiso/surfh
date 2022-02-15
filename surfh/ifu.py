# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

import sys
from typing import List, Sequence, Tuple

import numpy as np
from astropy.io import fits

array = np.ndarray


class Channel:
    """A channel with FOV, slit, spectral blurring and pce"""

    def __init__(
        self,
        beta_fov: float,
        alpha_fov: float,
        n_slit: int,
        w_blur: SpectralBlur,
        pce: array,
        wavel_axis: array,
        name: str = "",
    ):
        """

        wavel_axis is a regularly sampled wavelength axis
        """

        self.beta_fov = beta_fov
        self.alpha_fov = alpha_fov
        self.n_slit = n_slit

        self.w_blur = w_blur
        self.pce = pce
        self.wavel_axis = wavel_axis

        self.name = name

    @property
    def wavel_min(self):
        """The smallest sampled wavelength"""
        return self.wavel_axis[0]

    @property
    def wavel_max(self):
        """The largest sampled wavelength"""
        return self.wavel_axis[-1]

    @property
    def n_wavel(self):
        """The number of detector wavelength points"""
        return len(self.wavel_axis)

    @property
    def slit_beta_fov(self):
        """The width of slit"""
        return self.beta_fov / self.n_slit

    @property
    def wavel_axis(self):
        """The detector wavelength axis"""
        return np.linspace(self.wavel_min, self.wavel_max, self.n_wavel)

    @property
    def delta_wavel(self):
        """The detector wavelength sampling step"""
        return self.wavel_axis[1] - self.wavel_axis[0]

    def wslice(self, wavel_input_axis):
        """Return the measured wavelength within a selected channel"""
        return slice(
            np.where(wavel_input_axis <= self.wavel_min)[0][-1],
            np.where(wavel_input_axis >= self.wavel_max)[0][0],
        )

    def spectral_psf(self, beta_axis, wavel_input_axis):
        """Return spectral PSF for monochromatic punctual sources

        - The number of spatial positions inside a slit can be determined given
        the slit width and the beta_axis step size.

        - Beta_axis_slit refers to the beta_axis inside a slit, it is shifted
        around it's mean value, since the detector is calibrated to have the
        maximum of spectral psf at correct value of lambda for beta=0

        """
        beta_step = beta_axis[1] - beta_axis[0]
        beta_of_slits = np.arange(0, self.slit_beta_fov, beta_step)
        beta_of_slits -= np.mean(beta_of_slits)
        arcsec2micron = self.delta_wavel / beta_step
        if len(beta_axis) > 1:
            psfs = self.w_blur.psfs(
                self.wavel_axis,
                beta_of_slits,
                wavel_input_axis,
                arcsec2micron,
            )
        else:
            psfs = self.w_blur(self.wavel_axis, 0, wavel_input_axis)

        return psfs

    # def wslice_psf(self, beta_axis: array, wavel_input_axis: array):
    #     """Returns the index and the spectral PSF for the corresponding
    #     wavelength axis and beta_axis"""

    def slit_beta_fov_in_pixel(self, beta_axis: array):
        return int(np.round(self.slit_beta_fov / (beta_axis[1] - beta_axis[0])))

    def spatial_slice(
        self, alpha_axis: array, beta_axis: array, point: Sequence[int, int]
    ) -> Tuple[slice, slice]:
        """The spatial FOV for a given pointing"""

        alpha_step = alpha_axis[2] - alpha_axis[1]
        beta_step = beta_axis[2] - beta_axis[1]

        s_a = int(
            np.round((-self.alpha_fov / 2 + point[0] + alpha_axis[0]) / alpha_step)
        )
        e_a = int(
            np.round((+self.alpha_fov / 2 + point[0] + alpha_axis[0]) / alpha_step)
        )
        s_b = int(np.round((-self.beta_fov / 2 + point[1] + beta_axis[0]) / beta_step))
        e_b = int(np.round((+self.beta_fov / 2 + point[1] + beta_axis[0]) / beta_step))

        if s_b == e_b:
            raise ValueError("Null flux inside the slice with start_beta == end_beta")

        if (
            (s_a < alpha_axis[0] / alpha_step)
            or (e_a > alpha_axis[-1] / alpha_step)
            or (s_b < beta_axis[0] / beta_step)
            or (e_b > beta_axis[-1] / beta_step)
        ):
            raise ValueError("Observed FOV is bigger than input FOV")

        return (slice(s_a, e_a), slice(s_b, e_b))

    def all_spatial_slice(
        self,
        alpha_axis: array,
        beta_axis: array,
        pointings: Sequence[Sequence[int, int]],
    ) -> List[Tuple[slice, slice]]:
        """Return all spatial index for all pointing"""

        return [self.spatial_slice(alpha_axis, beta_axis, point) for point in pointings]


#%% \
class SpectralBlur:
    """A spectral response"""

    def __init__(self, grating_resolution: float):
        self.grating_resolution = grating_resolution
        #
        # the added margin serves for spectral PSF normalization. It is used in
        # private and eventually will not be taken into account at the projected
        # output
        self._n_margin = 10

    @property
    def grating_len(self):
        """The gratings len from given resolution R=λ/Δλ"""
        return 2 * 0.44245 / np.pi * self.grating_resolution

    def psfs(self, out_axis, beta, wavelength, scale: float = 1):
        """output array in [beta, wavelength, out_axis]"""
        beta = np.asarray(beta).reshape((-1, 1, 1))
        wavelength = np.asarray(wavelength).reshape((1, -1, 1))
        out_axis = np.asarray(out_axis).reshape((1, 1, -1))

        delta_w = wavelength[1] - wavelength[0]
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
        factor = np.pi * self.grating_len / wavelength
        out = (
            factor
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
