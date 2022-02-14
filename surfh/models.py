# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

from collections import namedtuple
from typing import List, Tuple

import numpy as np
import scipy.interpolate as sc
import udft
from mrs_functions import IFU_LMM
from sklearn import preprocessing
from sklearn.decomposition import PCA

array = np.ndarray

from . import ifu


def dft(in_array):
    return np.fft.rfftn(in_array, axes=(0, 1), norm="ortho")


def idft(in_array, shape):
    return np.fft.irfftn(in_array, axes=(0, 1), s=shape[:2], norm="orhto")


InputShape = namedtuple("InputShape", ["alpha", "beta", "wavel"])


class LMMDataModel:
    """Model a complete IFU"""

    def __init__(
        self,
        channels: List[ifu.Channel],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        pointings: List[Tuple[int, int]],
        n_components: int,
        sr_factors: List[int],
    ):
        self.channels = channels
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavel_axis = wavel_axis
        self.pointings = pointings
        self.n_components = n_components
        self.srf = sr_factors

        self.ishape = InputShape(len(alpha_axis), len(beta_axis), len(wavel_axis))

        self.spectral_psfs = [
            chan.spectral_psf(beta_axis, wavel_axis) for chan in channels
        ]
        self.spectral_slices = [chan.spectral_slice(wavel_axis) for chan in channels]
        self.spatial_indexs = [
            chan.all_spatial_index(alpha_axis, beta_axis, pointings)
            for chan in channels
        ]
        self.slit_width = [chan.slit_width_in_pixel(beta_axis) for chan in channels]

        self.pce = [chan.pce for chan in channels]

        self._otf_sr = [
            udft.ir2fr(np.ones((srf, 1)), self.ishape[:2])[np.newaxis, ...]
            for srf in self.srf
        ]

    def alpha_step(self):
        return self.alpha_axis[1] - self.alpha_axis[0]

    def beta_step(self):
        return self.beta_axis[1] - self.beta_axis[0]

    def precalc_spectral_output(self, components, psf_cube):
        """For each selected spectral component, and each selected channel
        with its corresponding slit width (say N pixels),
        this function precalculates N blurred
        and filtered spectra and associate each with a calculated psf cube"""

        spectral_output = []

        for idx, chan in enumerate(self.channels):

            # psfcube_chan = PSF_cube(chan.wavel_axis, self.alpha_step())
            spectral_output_chan = [
                Diffracted_psfcube(
                    components[p, :],
                    psf_cube[idx],
                    self.spectral_slices[idx],
                    self.spectral_psfs[idx],
                    self.slit_width[idx],
                    self.pce[idx],
                    self.ishape,
                )
                for p in range(self.n_components)
            ]

            spectral_output.append(spectral_output_chan)

        return spectral_output

    #%% \
    def forward(self, psf_cube, inarray):
        """Compute the forward and returns data for each channel"""
        if not np.iscomplexobj(inarray):
            # The input is in the spatial domain, so we calculate its FFT
            inarray = np.fft.rfftn(inarray, axes=(1, 2), norm="ortho")

        return [
            self.forward_channel(inarray, psf_cube[idx], idx)
            for idx, chan in enumerate(self.channels)
        ]

    def forward_channel(self, abundance, psf_cube, idx):
        """Retruns 4D array representing the data for a selected channel
         4D : (pointings, diffracted wavelength_detector, spatial_axis_detector, slit)

        Parameters
        ==========
        abundance : the abundance map corresponding to the selected spectral_component in Fourier space

        blured_spectrum: The PSF cube for a chosen channel and spectral component in the Fourier space

        slit_width : The slit width in pixels for one channel

        spatial_index: The different FOV for different pointongs

        otf_sr: Super resolution in the alpha direction vector taken into Fourier space

        sr_factor : Super resolution factor for a channel
        """

        shape = (self.ishape.alpha, self.ishape.beta)

        len_beta = int(
            np.round(
                (
                    self.spatial_indexs[idx][0][1].stop
                    - self.spatial_indexs[idx][0][1].start
                )
                / self.slit_width[idx]
            )
        )
        len_alpha = int(
            np.round(
                (
                    self.spatial_indexs[idx][0][0].stop
                    - self.spatial_indexs[idx][0][0].start
                )
                / self.srf[idx]
            )
        )
        out_channel = np.zeros(
            (
                len(self.spatial_indexs[idx]),
                self.channels[idx].n_wavel,
                len_alpha,
                len_beta,
            )
        )

        for slit in range(self.slit_width[idx]):
            otf_out = np.zeros_like(psf_cube[0][0], dtype=complex)
            for comp in range(self.n_components):
                otf_out += (
                    abundance[comp, np.newaxis, :, :]
                    * psf_cube[comp][slit]
                    * self._otf_sr[idx]
                )
            out = udft.irfftn(otf_out, axes=(1, 2), s=shape, norm="ortho")
            out_channel += spatial_indexing(
                out,
                slit,
                self.slit_width[idx],
                self.spatial_indexs[idx],
                len_beta,
                self.srf[idx],
            )
        return out_channel

    #%%
    def backward(self, psf_cube, shape, data):
        return udft.irfftn(
            self.backward_otf(psf_cube, data), axes=(1, 2), s=shape, norm="ortho"
        )

    def backward_otf(self, psf_cube, data):
        shape = (self.ishape.alpha, self.ishape.beta)
        back_otf_out = np.zeros(
            (self.n_components, psf_cube[0][0][0].shape[1], psf_cube[0][0][0].shape[2]),
            dtype=complex,
        )
        for idx, chan in enumerate(self.channels):
            for slit in range(self.slit_width[idx]):
                backward_index = backward_spatial_indexing_otf(
                    data[idx],
                    self.spatial_indexs[idx],
                    self.slit_width[idx],
                    slit,
                    self.srf[idx],
                    shape,
                )
                tmp_out = np.zeros(
                    (
                        self.n_components,
                        psf_cube[idx][0][0].shape[1],
                        psf_cube[idx][0][0].shape[2],
                    ),
                    dtype=complex,
                )
                for comp in range(self.n_components):
                    tmp_out[comp, ...] = np.sum(
                        psf_cube[idx][comp][slit].conj()
                        * self._otf_sr[idx].conj()
                        * backward_index,
                        axis=0,
                    )

                back_otf_out += tmp_out

        return back_otf_out

    def forward_backward_otf(self, spectral_output, abundance):
        return self.backward_otf(
            spectral_output, self.forward(spectral_output, abundance)
        )

    def forward_backward(self, spectral_output, shape, abundance):
        return self.backward(
            spectral_output, shape, self.forward(spectral_output, abundance)
        )

    def backward_coadd(self, data):
        shape = (self.ishape[0], self.ishape[1])
        out_coadd = []
        for idx, channel in enumerate(self.channels):
            wfilter = channel.wavel_filter.transmittance(channel.wavel_axis)
            tmp = (
                np.repeat(data[idx], self.slit_width[idx], axis=3)
                / self.slit_width[idx]
            )
            tmp = np.repeat(tmp, self.srf[idx], axis=2) / self.srf[idx]
            tmp = tmp / np.reshape(wfilter, [1, -1, 1, 1])
            backward_index = np.zeros((data[0].shape[1], shape[0], shape[1]))
            n_hit = np.zeros((data[0].shape[1], shape[0], shape[1]))

            for si in range(len(self.spatial_indexs[idx])):
                backward_index[
                    :,
                    self.spatial_indexs[idx][si][0],
                    self.spatial_indexs[idx][si][1],
                ] += tmp[si, :, :, :]
                n_hit[
                    :,
                    self.spatial_indexs[idx][si][0],
                    self.spatial_indexs[idx][si][1],
                ] += 1
            out_coadd_chan = np.where(n_hit == 0, 0, backward_index / n_hit)
            out_coadd.append(out_coadd_chan)
        return out_coadd

    def shift_and_coadd(self, data, wavel_axis, ifu):
        out_coadd = np.zeros((self.ishape[2], self.ishape[0], self.ishape[1]))
        n_hit = np.zeros_like(out_coadd)
        for idx, channel in enumerate(self.channels):
            spatial_interpolation = self.spatial_interpolation_for_coadd(
                data, idx, channel, ifu
            )
            inter = sc.interp1d(
                np.linspace(
                    0, spatial_interpolation.shape[0], spatial_interpolation.shape[0]
                ),
                spatial_interpolation,
                axis=0,
                kind="nearest",
            )
            spectral_interpolation = inter(
                np.linspace(
                    0,
                    spatial_interpolation.shape[0],
                    wavel_axis[ifu.spectral_slices[idx]].shape[0],
                )
            )
            fov = IFU_LMM.fov(ifu.spatial_indexs[idx])
            out_coadd[ifu.spectral_slices[idx], :, :] += spectral_interpolation
            n_hit[ifu.spectral_slices[idx], fov[0] : fov[1], fov[2] : fov[3]] += 1
        out_coadd_normalized = np.where(n_hit == 0, 0, out_coadd / n_hit)
        return out_coadd_normalized

    def spatial_interpolation_for_coadd(self, data, idx, channel, ifu):

        shape = (self.ishape[0], self.ishape[1])

        wfilter = channel.wavel_filter.transmittance(channel.wavel_axis)

        tmp = np.repeat(data[idx], self.slit_width[idx], axis=3) / self.slit_width[idx]
        tmp = np.repeat(tmp, self.srf[idx], axis=2) / self.srf[idx]
        tmp = tmp / np.reshape(wfilter, [1, -1, 1, 1])

        backward_index = np.zeros((data[0].shape[1], shape[0], shape[1]))
        n_hit = np.zeros((data[0].shape[1], shape[0], shape[1]))
        for si in range(len(self.spatial_indexs[idx])):
            b = backward_index[
                :,
                self.spatial_indexs[idx][si][0],
                self.spatial_indexs[idx][si][1],
            ]
            b[:, :, : tmp.shape[-1]] += tmp[si, :, :, :]

            n_hit[
                :,
                self.spatial_indexs[idx][si][0],
                self.spatial_indexs[idx][si][1],
            ] += 1
        out_coadd_chan = np.where(n_hit == 0, 0, backward_index / n_hit)

        return out_coadd_chan


#%%


def Diffracted_psfcube(
    components, psf_cube, spectral_index, psfs, slit_width, wfilter, in_shape
):
    shape = (in_shape[0], in_shape[1])
    diff_otf = []
    for idx in range(slit_width):
        diffracted_component_channel = (
            spectral_diffraction(
                components, spectral_index, psfs[idx, :, :], slit_width, wfilter
            )[:, np.newaxis, np.newaxis]
            * psf_cube
        )
        diff_otf_single = udft.ir2fr(diffracted_component_channel, shape)
        diff_otf.append(diff_otf_single)

    return diff_otf


def spectral_diffraction(components, spectral_index, psfs, slit_width, wfilter):
    filtered_comp = spectral_filter(components[spectral_index], wfilter)
    return sum(np.dot(filtered_comp[np.newaxis, :], psfs))


def spectral_diffraction_coadd(components, spectral_index, psfs, slit_width, wfilter):
    out_spectrum = []
    for idx in range(slit_width):
        filtered_comp = spectral_filter(components[spectral_index], wfilter)
        blurred_spectrum = sum(np.dot(filtered_comp[np.newaxis, :], psfs[idx, :, :]))
        out_spectrum.append(blurred_spectrum)
    return out_spectrum


def spectral_filter(components, wfilter):
    return components * wfilter


def spectral_filter_coadd(components, wfilter):
    return components / wfilter


#%%
def spatial_indexing(array, slit, slit_width, spatial_index, len_beta, srf):
    """Select the observed FOV within a pointing and selects the right spatial position
    inside a slice"""
    out_indexed = []
    for si in spatial_index:
        indexed_out = array[:, si[0], si[1]][:, ::srf, slit::slit_width]
        out_indexed.append(indexed_out[:, :, :len_beta])
    return np.asarray(out_indexed)


def backward_spatial_indexing(data, spatial_index, slit_width, slit, srf, shape):
    out_transp = np.zeros((data.shape[1], shape[0], shape[1]))
    for s_i in range(len(spatial_index)):
        out_transp[:, spatial_index[s_i][0], spatial_index[s_i][1]][
            :, ::srf, slit::slit_width
        ] += data[s_i, :, :, :]
    return out_transp


def backward_spatial_indexing_otf(data, spatial_index, slit_width, slit, srf, shape):
    return np.fft.rfftn(
        backward_spatial_indexing(data, spatial_index, slit_width, slit, srf, shape),
        axes=(1, 2),
        norm="ortho",
    )


def backward_spatial_coadd(data, spatial_index, slit_width, slit, srf, shape):
    n_hit = np.zeros((data[0].shape[0], shape[0], shape[1]))
    len_alpha = int(np.round((spatial_index[0][0].stop - spatial_index[0][0].start)))

    tmp = np.repeat(data, srf, axis=2) / srf
    out_transp = np.zeros((data.shape[1], shape[0], shape[1]))

    for s_i in range(len(spatial_index)):
        out_transp[:, spatial_index[s_i][0], spatial_index[s_i][1]][
            :, :, slit::slit_width
        ] += tmp[s_i, :, :len_alpha, :]
        n_hit[:, spatial_index[s_i][0], spatial_index[s_i][1]] += 1
    return np.where(n_hit == 0, 0, (out_transp) / n_hit)
    # return out_transp , n_hit


#%%
def PSF_cube(wavel_axis, step, D=6.5):
    x_axis = np.arange(0, 20)
    y_axis = x_axis[:, np.newaxis]
    x0 = y0 = len(x_axis) // 2
    Psf_cube = []

    for wavel in wavel_axis:
        FWHM_arcsec = (wavel * 1e-6 / D) * 206265  # from rad to arcsec
        sigma = FWHM_arcsec / (step * 2.354)  # sigma in pixels
        PSF = (
            1.0
            / (2.0 * np.pi * sigma**2)
            * np.exp(
                -(
                    (x_axis - x0) ** 2 / (2 * sigma**2)
                    + (y_axis - y0) ** 2 / (2 * sigma**2)
                )
            )
        )
        Psf_cube.append(PSF)
    # return np.rollaxis(np.asarray(Psf_cube), 0 , 3)
    return np.asanyarray(Psf_cube)


def precalc_otf(wavel_axis, alpha_step, shape, components):
    spatial_otf = PSF_cube(wavel_axis, alpha_step)
    otf_spectrum = udft.ir2fr(
        spatial_otf[np.newaxis, ...] * components[:, :, np.newaxis, np.newaxis], shape
    )
    return otf_spectrum


def spatial_sampling(in_array, sampling_factor):
    otf_sr = udft.ir2fr(
        np.ones((sampling_factor, sampling_factor)) / sampling_factor**2,
        in_array.shape,
    )
    LR_map = LR_map = np.fft.irfft2(np.fft.rfft2(in_array) * otf_sr, in_array.shape)
    return LR_map[::sampling_factor, ::sampling_factor]


def dim_reduction(in_array, n_components):
    pca = PCA(n_components=n_components)
    in_array2D = in_array.reshape(-1, in_array.shape[-1])
    new_array = pca.fit(in_array2D)
    axis_projection = new_array.transform(in_array2D)
    components = new_array.components_
    abundance = axis_projection.reshape((in_array.shape[0], in_array.shape[1], -1))
    return components, abundance


def full_cube(abundance, components):
    return np.moveaxis(np.dot(np.moveaxis(abundance, 0, -1), components), -1, 0)


def Full_optics(wavel_axis, step, shape):
    return udft.ir2fr(PSF_cube(wavel_axis, step), shape)
