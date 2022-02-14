# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 28 11:03:52 2016

@author: amine
"""

import functools
import itertools as it
import time
from time import time as timer

import matplotlib.patches as patches
import matplotlib.ticker as ticker

# import numba as nb
import numpy as np
import numpy.random as npr

# import pysynphot as synph
# import webbpsf
# from astropy.io import fits
from matplotlib import pyplot as plt
from numpy.random import randn, seed
from scipy.integrate import trapz
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
#from xlrd import open_workbook

#from librairie.improcessing import udeconv
#from librairie.optim import conj_grad
#from librairie.udft import ir2fr, udft2, uidft2, uirdft2, urdft2

# pylint: disable=E1101

# %% import useful librairies

# import glob

# from matplotlib.colors import LogNorm

# ==============================================================================
# Mid InfraRed Instrument (MIRI)
# ==============================================================================

# Definition of constants
EPS = 2.2204e-31


def load_miri():
    """  wavelength are in microns    """
    #    miri = webbpsf.MIRI()

    miri_filters = dict(
        name=[
            "F560W",
            "F770W",
            "F1000W",
            "F1130W",
            "F1280W",
            "F1500W",
            "F1800W",
            "F2100W",
            "F2550W",
        ],
        total_number=9,
        width=np.array([1.20, 2.20, 2.20, 0.70, 2.40, 3.00, 3.00, 5.00, 4.00]),
        central_wave=np.array(
            [5.60, 7.70, 10.00, 11.30, 12.80, 15.50, 18.00, 21.00, 25.50]
        ),
    )

    miri_param = dict(pixelscale=0.11, omega_pix=0.0121, tau_eol=0.8, tau_tel=0.88)
    psf_param = dict(oversamp=1, fov=5)
    psf_param.update(
        dict(
            size=int(
                round(psf_param["fov"] / miri_param["pixelscale"])
                * psf_param["oversamp"]
            )
        )
    )

    return miri_filters, miri_param, psf_param


# %% Point Spread Function (PSF)


def compute_monochromatic_psfs(wave_filter, psf_fov, psf_size):
    """
    https://pythonhosted.org/webbpsf/webbpsf.html#psf-normalization
    This function use webbpsf tool to simulate monochromatic psf
    how to use :
    psfOverSamp = 1 # detector resolution
    psfFov = 5      # arcsec
    psf_size = int(round(psfFov/MIRI.pixelscale)*psfOverSamp)
    psfPath = "PSFs/detectorRes/"
    #psfs_monochromatic = compute_monochromatic_psfs(lamAllMIRI, psfFov, psf_size)
    np.save('PSFs/detectorRes/psfMonochromatic_AlainAbergelWave_crop.npy', psf_obj)
    psf = phd.compute_monochromatic_psfs(wave, psf_param['fov'], psf_param['size'])
    """
    miri = webbpsf.MIRI()
    psf_number = len(wave_filter)
    psfs_monoch = np.zeros((psf_number, psf_size, psf_size))
    for i in range(psf_number):
        #        print(wave_filter[i]*1e-6)
        miri.options["output_mode"] = "detector sampled"
        psf_file = miri.calcPSF(
            monochromatic=wave_filter[i] * 1e-6, fov_arcsec=psf_fov, rebin=True
        )
        psfs_monoch[i] = psf_file[0].data
    return psfs_monoch


# =============================================================================


def compute_broadband_psfs(src, wave_src, pce_miri, psfs_monochromatic):
    """psfs_broadband = compute_broadband_psfs(pce_miri, psfs_monochromatic)"""

    m, n = pce_miri.shape
    #    dummy_wave_n, psf_size, psf_size = psfs_monochromatic.shape
    l, psf_size, psf_size = psfs_monochromatic.shape

    psfs_broadband = np.zeros((n - 1, psf_size, psf_size), dtype=np.float32)

    # Resampling the src using linear interpolation
    lambda_interp = pce_miri[:, 0]
    src_interp = np.interp(lambda_interp, wave_src, src)

    for filt in range(n - 1):
        #        print("filtre %d"%(filt+1))
        # extract wavelengths of filt using PCE
        pce_channel = pce_miri[:, filt + 1]
        index = np.where(pce_channel > 0)
        lambda_channel = pce_miri[index[0], 0]

        tmp = (
            pce_channel[index[0]].reshape(-1, 1, 1)
            * src_interp[index[0]].reshape(-1, 1, 1)
            * psfs_monochromatic[index[0]]
        )
        psfs_broadband[filt] = trapz(tmp, x=lambda_channel, axis=0)
    return psfs_broadband


# =============================================================================


def psf_pad_circ_shift(psf, out_size):
    """ Pad the PSF to outSize"""
    pad_size = np.array(out_size) - np.array(psf.shape)
    psf_padded = np.lib.pad(psf, ((0, pad_size[0]), (0, pad_size[1])), "constant")
    # Circularly shift otf so that the "center" of the PSF is at the (0, 0)
    # element of the array
    psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[0] / 2.0)), axis=0)
    psf_shifted = np.roll(psf_padded, -int(np.floor(psf.shape[1] / 2.0)), axis=1)
    return psf_shifted


# =============================================================================


def pce_function(file_name):
    """ ... """
    wbook = open_workbook(filename=file_name)
    sheet = wbook.sheet_by_index(0)
    pce_xls = [
        [sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)
    ]
    pce_data = np.asarray(pce_xls)
    pce_miri = pce_data[2:-1, :]
    pce_miri = pce_miri.astype(float)

    #    # clip the spectral axis : \lambda \in [4.5, 28.8] microns
    #    pce_miri[pce_miri[:, :] < 0.001] = 0.0
    #    dummy_tmp, idx = find_interval_of_x(wave=pce_miri[:, 0], wave_x=4.5)
    #    pce_miri = pce_miri[idx[-1]:-1, :]

    # nLambda, nFeature = PCE_MIRI.shape
    # deltaLam = (lamAllMIRI[-1]-lamAllMIRI[0])/(len(lamAllMIRI)-1) #d\lamda

    return np.swapaxes(pce_miri, 0, 1)


# =============================================================================


def psf2otf(psf, shape, real=False):
    """
    OTF = psf2otf(PSF) computes the fast Fourier transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer function
    array, OTF, that is not influenced by the PSF off-centering
    """
    if real:
        psf_padcircshift = psf_pad_circ_shift(psf, [shape[0], shape[1]])
        otf = np.fft.rfft2(psf_padcircshift)
    else:
        psf_padcircshift = psf_pad_circ_shift(psf, [shape[0], shape[1]])
        otf = np.fft.fft2(psf_padcircshift)
    return otf


# ==================================================================================================
# Object related functions
# ==================================================================================================


def create_src_spectrum(param):
    """source http://pysynphot.readthedocs.io/en/latest/spectrum.html#
    pysynphot-flat-spec
    http://nullege.com/codes/show/src@p@y@pysynphot-0.9.5@test@test_
    analytic_spectrum.py/143/pysynphot.BlackBody.sample
    sp = S.ArraySpectrum(w, f, name='MySource', waveunits='microns')"""
    # flam = erg cm^-2 s^-1 Ang^-1

    if param["srcType"] == "PowerLaw":
        return synph.PowerLaw(
            param["wave_ref_microns"],
            param["pl_ind"],
            waveunits=param["waveunit"],
            fluxunits=param["fluxunit"],
        )

    elif param["srcType"] == "Flat":
        return synph.FlatSpectrum(
            param["amplitude"], waveunits=param["waveunit"], fluxunits=param["fluxunit"]
        )

    elif param["srcType"] == "blackbody":
        return synph.BlackBody(param["temp_kelvin"])

    elif param["srcType"] == "Gaussian":
        return synph.GaussianSource(
            param["f_tot"],
            param["wave_mean"],
            param["fwhm"],
            waveunits=param["waveunit"],
            fluxunits=param["fluxunit"],
        )
    else:
        print("Error:please check the correct name of the spectrum type")
    #        \textnormal{flux} = (\lambda_ / \lambda_{0} )^{-\alpha}
    # wave_target = wave_microns*1e4 # 1 Angstrom = 1e-4 microns
    # \sigma = \frac{\textnormal{FWHM}}{2 \sqrt{2 ln 2}}
    #        A = \frac{f_{\textnormal{tot}}}{\sqrt{2 \pi} \; \sigma}
    #        \textnormal{flux} = A  /  exp(\frac{(x - x_{0})^{2}}{2 \sigma^{2}})


# bb = createSrcSpectrum('Flat', WAVE_AXIS_MIRI, 5500)
#    plt.figure()
#    plt.plot(WAVE_AXIS_MIRI, bb)
#    plt.show()

# changement d'unité
#    sp = S.GaussianSource(18.3, 18000, 2000)
#    ff=trapz(sp.flux, x=sp.wave, axis=0)
#    plt.figure()
#    plt.plot(sp.wave, sp.flux)
#
#    sp.convert('nm')
#    sp.convert('flam')
#    ff2=trapz(sp.flux, x=sp.wave, axis=0)
#    plt.plot(sp.wave, sp.flux,'r')


# =============================================================================
# computation funtions
# =============================================================================

# ==================================================================================================


def conv_fft(obj, psf):
    """This function compute the convolution between x and h in the fourier domain"""
    otf = psf_pad_circ_shift(psf, [obj.shape[0], obj.shape[1]])
    return np.real(
        uidft2(
            udft2(obj, s=[obj.shape[0], obj.shape[1]])
            * udft2(otf, s=[obj.shape[0], obj.shape[1]])
        )
    )


# ==================================================================================================
def gaussian_2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    size_m, size_n = [(ss - 1.0) / 2.0 for ss in shape]
    grid_y, grid_x = np.ogrid[-size_m : size_m + 1, -size_n : size_n + 1]
    kernel = np.exp(-(grid_x * grid_x + grid_y * grid_y) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel = kernel / kernel_sum
    return kernel


# ==================================================================================================
# @nb.jit
def interp2d(wave_new, wave_obj, obj):
    """ comment """
    __, HEIGHT, WIDTH = obj.shape
    obj_interp = np.zeros((len(wave_new), HEIGHT, WIDTH), dtype=np.float32)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            obj_interp[:, i, j] = np.interp(wave_new, wave_obj, obj[:, i, j])
    return obj_interp


# preconditionning
def M(H_freq, obj_freq):
    """ precondionner """
    H_freq_abs_2 = 1 / np.abs(np.ones_like(H_freq)) ** 2
    M_x_freq = np.sum(np.multiply(H_freq_abs_2, obj_freq), axis=0)
    return M_x_freq


# ==================================================================================================
# Display
# ==================================================================================================


def display_simulated_data(y, in_param, miri_filter, dic):
    """..."""
    fig = plt.figure(figsize=cm2inch(12, 12))
    for f in range(miri_filter["total_number"]):
        plt.subplot(3, 3, f + 1)
        im = plt.imshow(y[f])
        plt.title("{0:s}".format(miri_filter["name"][f]))
        plt.axis("off")

        if f != 6:
            plt.colorbar()
        if f == 6:
            plt.axis("on")
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel(dic["ylabel"], fontsize=18)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(str(dic["dir_result"] + dic["fname"]), bbox_inches="tight")
    plt.close(fig)


# ==================================================================================================


def plot_datacube(datacube, vmin=None, vmax=None, cmap=None, norm=None, colorbar=None):
    """ display cube """

    def fmt(x):
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        return r"${} \times 10^{{{}}}$".format(a, b)

    n, __, __ = datacube.shape

    if not (vmin and vmax):
        vmin = datacube.min()
        vmax = datacube.max()
    fig = plt.figure()
    plt.clf()
    fig.set_size_inches(8.5, 8.5, forward=True)
    for i in range(min(n, 9)):
        plt.subplot(3, 3, i + 1)
        m = plt.imshow(datacube[i], vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)

        if i == 0:
            # plt.xlabel(r'arcsec', fontsize=18)
            # plt.ylabel(r'arcsec', fontsize=18)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            if colorbar:
                plt.colorbar(m, format=ticker.FuncFormatter(fmt))
            else:
                plt.colorbar()

        elif i != 0:
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()


# ==================================================================================================


def display_object_slice(folder, wave_obj, obj_pad, obj_psc, Slice):
    fig = plt.figure(11)
    plt.clf()
    fig.set_size_inches(12, 13, forward=True)

    ax = (fig.add_subplot(2, 2, 1),)
    plt.imshow(obj_pad[Slice], cmap="gray")
    plt.colorbar(),
    plt.title("First slice : 0.5''/pix, \n Flux= %2.2e MJy/ster" % flux_obj_pad[Slice])
    plt.xlabel('200 pixel=100"')
    plt.ylabel('200 pixel=100"')

    ax = plt.subplot(2, 2, 2)
    plt.imshow(obj_psc[Slice], cmap="gray")
    plt.colorbar()
    plt.title("First slice : 0.11''/pix, \n Flux= %2.2e MJy/ster" % flux_obj_psc[Slice])
    plt.xlabel(r'910 pixel=100"')
    plt.ylabel(r'910 pixel=100"')

    rect = patches.Rectangle((3, 3), 512, 512, lw=2, edgecolor="r", facecolor="none")
    ax.add_patch(rect)
    color = {"cl1": "#e66101", "cl2": "#fdb863", "cl3": "#b2abd2", "cl4": "#5e3c99"}
    plt.subplot(2, 1, 2),
    plt.plot(
        wave_obj,
        obj_pad[:, 20, 20] / obj_pad[:, 20, 20].max(),
        color["cl1"],
        label=r"pix$_{20,20}$",
    )
    plt.plot(
        wave_obj,
        obj_pad[:, 20, 40] / obj_pad[:, 20, 40].max(),
        color["cl2"],
        label=r"pix$_{20,40}$",
    )
    plt.plot(
        wave_obj,
        obj_pad[:, 20, 60] / obj_pad[:, 20, 60].max(),
        color["cl3"],
        label=r"pix$_{20,60}$",
    )
    plt.plot(
        wave_obj,
        obj_pad[:, 20, 80] / obj_pad[:, 20, 80].max(),
        color["cl4"],
        label=r"pix$_{20,80}$",
    )
    plt.plot(
        wave_obj,
        obj_pad[:, 20, 200] / obj_pad[:, 20, 200].max(),
        label=r"pix$_{20,80}$",
    )
    plt.ylabel(r"Photon Flux [MJy/Ster]")
    plt.legend()

    plt.savefig(folder + "/synthetic_object.pdf", bbox_inches="tight")
    plt.close(fig)


# ==================================================================================================


def disp_imshow(obj, disp_obj):
    plt.imshow(obj, vmin=disp_obj["vmin"], vmax=disp_obj["vmax"], cmap="gray")
    plt.title(disp_obj["title"], fontsize=disp_obj["fontsize"])
    plt.xlabel(disp_obj["xlabel"])
    plt.ylabel(disp_obj["ylabel"])
    cbar = plt.colorbar()
    cbar.set_label(disp_obj["cbar_label"])


# ==================================================================================================


def disp_plot(wave_obj, obj, disp_obj):
    plt.plot(wave_obj, obj, label=disp_obj["label"])
    plt.title(disp_obj["title"])
    plt.xlabel(disp_obj["xlabel"])
    plt.ylabel(disp_obj["ylabel"])
    plt.autoscale(enable=True, axis="x", tight=True)


# ==================================================================================================


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


# %%===========================================================================
# Simulate object
# =============================================================================


def get_object_from_cube(x_restored, wave, wave_axis_cent, wave_axis_band):
    """this function return a 2D object from a given : cube-object and a
    wavelength, the cube-object is supposed to be linear spectrally"""
    obj_n, __, __ = x_restored.shape

    phi_0 = np.asarray(
        [(x_restored[i + 1] + x_restored[i]) / 2 for i in range(obj_n - 1)]
    )
    phi_1 = np.asarray(
        [
            (x_restored[i + 1] - x_restored[i])
            / (wave_axis_band[i + 1] - wave_axis_band[i])
            for i in range(obj_n - 1)
        ]
    )

    __, wave_idx = find_interval_of_x(wave_axis_band, wave)

    return [
        phi_0[wave_idx[0]] + phi_1[wave_idx[0]] * (wave - wave_axis_cent[wave_idx[0]]),
        phi_0[wave_idx[0]],
    ]


#    return phi_0[wave_idx[0]]+phi_1[wave_idx[0]]*(wave-wave_axis_cent[wave_idx[0]])

# =============================================================================


def find_interval_of_x(wave, wave_x):
    """This function allow us to locate and return the index of wave_x in wave ,
    wave is a vector and wave_x is a scalar, wave must contain wave_x"""
    if wave[0] <= wave_x <= wave[-1]:
        index = np.where(wave == wave_x)[0]
        if len(index):  # wave_x is an element of wave
            return [wave_x, index]
        else:  # wave_x is not an element of wave
            tmp1 = np.where(wave < wave_x)[0]
            tmp2 = np.where(wave > wave_x)[0]
            index = [tmp1.max(), tmp2.min()]
            return [wave[index], index]
    else:
        print("Error: the array wave must include the value wave_x")


# =============================================================================


def create_cube_linear_sources(obj_shape, wave_axis, features, kernel, position):
    """ .. """
    obj_n, obj_height, obj_width = obj_shape
    n_source = len(position)
    obj_true = np.zeros((obj_n, obj_height, obj_width), dtype=np.float32)
    linear_src = np.zeros((n_source, obj_n), dtype=np.float32)
    slope, level = features
    lam_c = (wave_axis[-1] + wave_axis[0]) / 2
    linear_src = [
        slope[idx] * (wave_axis - lam_c) + level[idx] for idx, val in enumerate(slope)
    ]
    HEIGHT_C = obj_height // 2 + 1
    for idx_s in range(n_source):
        obj_true[:, HEIGHT_C, position[idx_s]] = linear_src[idx_s]
    if np.any(kernel):
        conv = lambda src, h: np.real(
            uirdft2(urdft2(src) * ir2fr(h, src.shape, real=True))
        )
        for i in range(obj_n):
            obj_true[i] = conv(obj_true[i], kernel)
    return obj_true


# =============================================================================


def create_cube_linear_source(obj_shape, wave_axis, features, kernel=None):
    """Create a cube for an object containing one source with a linear
    spectrum led by features parameter"""
    obj_n, obj_height, obj_width = obj_shape
    obj_true = np.zeros((obj_n, obj_height, obj_width), dtype=np.float32)
    linear_src = np.zeros(obj_n, dtype=np.float32)
    slope, level = features

    lam_c = (wave_axis[-1] + wave_axis[0]) / 2
    linear_src = slope * (wave_axis - lam_c) + level

    obj_true[:, obj_height // 2 + 1, obj_width // 2 + 1] = linear_src

    if np.any(kernel):
        conv = lambda src, h: np.real(
            uirdft2(urdft2(src) * ir2fr(h, src.shape, real=True))
        )
        for b in range(obj_n):
            obj_true[b] = conv(obj_true[b], kernel)
    return obj_true


# =============================================================================


def create_cube_from_spectrum(shape, spectrum, kernel=None):
    """Create a cube for an object containing one source with a linear
    spectrum led by features parameter"""
    obj_n, height, width = shape
    obj_true = np.zeros((obj_n, height, width), dtype=np.float32)

    height_center = height // 2 + 1
    width_center = width // 2 + 1
    obj_true[:, height_center, width_center] = spectrum

    if np.any(kernel):
        conv = lambda src, h: np.real(
            uirdft2(urdft2(src) * ir2fr(h, src.shape, real=True))
        )
        obj_true = np.asarray([conv(obj_true[b], kernel) for b in range(obj_n)])
    return obj_true


# %%
def dirac(obj, val):
    """ fonction dirac dicrete; x and t are scalars"""
    if obj == val:
        out = 1
    else:
        out = 0
    return out


# =============================================================================


def prox_gamma_g(obj, gamma, mu_reg):
    """  soft-thresholding """
    return obj - obj / np.max(np.abs(obj) / (mu_reg * gamma), 1)


# %% Direct Model


def instrument_model(wave, pce, psf, obj, in_param):
    """This instrument model take into account the convolution with the spectral-variant PSF and a spectral
    integration"""
    N_lambda, height, width = obj.shape
    if in_param["conv"] == False:
        prod_conv = obj
    else:
        prod_conv = np.asarray(
            [
                uirdft2(imf * ir2fr(psf[idx], [height, width]))
                for idx, imf in enumerate(urdft2(obj))
            ]
        )

    return np.asarray(
        [
            trapz((pce[f + 1]).reshape(-1, 1, 1) * prod_conv, x=wave, axis=0)
            for f in range(in_param["n_f"])
        ]
    )


# =============================================================================


def direct_model(filters, obj_f):
    """This function return the output of the forward model JWST/MIRI in
    frequency-domain"""
    data_freq = np.asarray([np.sum(filt * obj_f, axis=0) for filt in filters])
    data = np.real(uirdft2(data_freq))
    return data, data_freq


# =============================================================================


def exec_direct_model_1(wave_miri, psf_obj, pce_obj, wave_obj, in_param):
    """..."""

    # if in_param['source_type'] == "horsehead":
    #     spec_samp = input("Enter sampling type: 'uniform' or 'non_uniform' ").lower()
    #     assert spec_samp == 'uniform' or spec_samp == 'non_uniform', 'Please check the entry'
    # else:
    #     spec_samp = "uniform"
    def calc_psf_int_1(psf, psf_wave, wave_axis_band, pce_miri):
        """
        This function computes convolution kernels for each filter,
        when considering a multiband-multifilter forward model
        """
        n, m = pce_miri.shape
        n_filt = n - 1
        n_band = len(wave_axis_band) - 1
        l, k = psf.shape[1:]
        h0_ = np.zeros((n_filt, n_band, l, k), dtype=np.float32)
        h1_ = np.zeros((n_filt, n_band, l, k), dtype=np.float32)

        wave_axis_cent = np.zeros(n_band)
        for b in range(n_band):
            idx = np.where(
                np.logical_and(
                    (psf_wave <= wave_axis_band[b + 1]), (wave_axis_band[b] <= psf_wave)
                )
                == True
            )
            wave_band = psf_wave[idx]
            psf_band = psf[idx]
            #        print(wave_band)

            if (wave_band[-1] - wave_band[0]) != 0:
                #            wave_axis_cent[b] = (wave_band[-1] + wave_band[0])/2
                #            g_0 = 0.5 - (wave_band-wave_axis_cent[b]) /(wave_band[-1]-wave_band[0])
                #            g_1 = 0.5 + (wave_band-wave_axis_cent[b]) /(wave_band[-1]-wave_band[0])
                g_0 = (wave_band[-1] - wave_band) / (wave_band[-1] - wave_band[0])
                g_1 = (wave_band - wave_band[0]) / (wave_band[-1] - wave_band[0])
            else:
                #            wave_axis_cent[b] = (wave_band[-1] + wave_band[0])/2
                g_0 = 0
                g_1 = 0
            for f in range(n_filt):
                pce = pce_miri[f + 1]
                pce_band = pce[idx]
                temp_0 = (pce_band * g_0).reshape(-1, 1, 1) * psf_band
                temp_1 = (pce_band * g_1).reshape(-1, 1, 1) * psf_band
                h0_[f, b] = trapz(temp_0, x=wave_band, axis=0)
                h1_[f, b] = trapz(temp_1, x=wave_band, axis=0)

        psf_int = np.zeros((n_filt, n_band + 1, l, k), dtype=np.float32)
        for f in range(n_filt):
            for m in range(n_band + 1):
                if m == 0:
                    psf_int[f, m] = h0_[f, m] + 0
                elif 1 <= m <= n_band - 1:
                    psf_int[f, m] = h0_[f, m] + h1_[f, m - 1]
                elif m == n_band:
                    psf_int[f, m] = 0 + h1_[f, m - 1]
        return psf_int, wave_axis_cent

    spectral_sampling = "uniform"
    in_param.update({"spec_samp": spectral_sampling})

    if in_param["spec_samp"] == "uniform":
        wave_band = np.linspace(4.7, wave_miri[-1], in_param["n_samp"])
    elif in_param["spec_samp"] == "non_uniform":
        wave_band = np.concatenate(
            (np.linspace(4.7, 12.64, in_param["n_samp"]), [13.68, wave_miri[-1]])
        )
    else:
        raise ValueError("Please enter check the entry : 'uniform' or 'non_uniform' ")

    print(
        r"{0:s} spectral sampling with {1:d} samples".format(
            in_param["spec_samp"], in_param["n_samp"]
        )
    )
    n_obj = len(wave_band)

    # Define operator H
    h_int, wave_axis_cent = calc_psf_int_1(psf_obj, wave_obj, wave_band, pce_obj)
    H_freq = np.asarray(
        [
            [
                ir2fr(h_int[f, m], shape=(in_param["H"], in_param["W"]), real=True)
                for m in range(n_obj)
            ]
            for f in range(in_param["n_f"])
        ]
    )
    return H_freq, wave_band, n_obj, h_int


# =============================================================================
# @nb.jit
def calc_g0_g1(lam):
    lam_1 = lam[0]
    lam_2 = lam[-1]

    if (lam_2 - lam_1) != 0:
        g_0 = (lam_2 - lam) / (lam_2 - lam_1)
        g_1 = (lam - lam_1) / (lam_2 - lam_1)
    else:
        g_0 = 0
        g_1 = 0
    return g_0, g_1


def forward_model_1(psf_obj, pce_obj, wave_band, wave_obj, in_param):
    """..."""

    def calc_psf_int_1(psf_obj, pce_obj, t, T):
        """
        This function computes convolution kernels for each filter
        using multiwavelength forward model
        T is the knot vector
        """
        n, __ = pce_obj.shape
        P = in_param["P"]
        M = len(T)
        n_band = M - 1
        l, k = psf_obj.shape[1:]
        h0_ = np.zeros((P, n_band, l, k), dtype=np.float32)
        h1_ = np.zeros((P, n_band, l, k), dtype=np.float32)
        h_int = np.zeros((P, M, l, k), dtype=np.float32)

        for b in range(n_band):
            # get index of wavelength within linear piece
            idx = np.where(np.logical_and((t <= T[b + 1]), (T[b] <= t)) == True)
            psf_band = psf_obj[idx]

            # compute g0 and g1
            lam = t[idx]
            g_0, g_1 = calc_g0_g1(lam)

            for p in range(P):
                pce_band = pce_obj[p + 1][idx]
                temp_0 = (pce_band * g_0).reshape(-1, 1, 1) * psf_band
                temp_1 = (pce_band * g_1).reshape(-1, 1, 1) * psf_band
                h0_[p, b] = trapz(temp_0, x=lam, axis=0)
                h1_[p, b] = trapz(temp_1, x=lam, axis=0)

        for f, m in it.product(range(P), range(M)):
            if m == 0:
                h_int[f, m] = h0_[f, m] + 0.0
            elif 1 <= m <= n_band - 1:
                h_int[f, m] = h0_[f, m] + h1_[f, m - 1]
            elif m == n_band:
                h_int[f, m] = 0.0 + h1_[f, m - 1]
        return h_int

    # Define operator H
    n_lam_obj = len(wave_band)
    h_int = calc_psf_int_1(psf_obj, pce_obj, wave_obj, wave_band)
    H_int_freq = np.asarray(
        [
            [
                ir2fr(h_int[f, m], shape=(in_param["H"], in_param["W"]), real=True)
                for m in range(n_lam_obj)
            ]
            for f in range(in_param["P"])
        ]
    )

    return H_int_freq, h_int


# =============================================================================


def forward_model_2(psf_obj, pce_obj, wave_obj, S_n, in_param):
    """..."""

    def calc_psf_int_2(psf_obj, pce_obj, wave_obj, S_n, in_param):
        """    This function computes the matrix h of the direct model ...    """
        l, k = psf_obj.shape[1:]
        n_s, n_lam = S_n.shape
        h_int = np.zeros((in_param["n_f"], n_s, l, k))
        pce_obj = pce_obj[1:]
        for f, pce in enumerate(pce_obj):
            for n, S in enumerate(S_n):
                temp = (pce * S).reshape(-1, 1, 1) * psf_obj
                h_int[f, n] = trapz(temp, x=wave_obj, axis=0)
        return h_int

    # Define operator H
    n_f = in_param["n_f"]
    n_s, __ = S_n.shape
    h_int = calc_psf_int_2(psf_obj, pce_obj, wave_obj, S_n, in_param)

    height = in_param["H"]
    width = in_param["W"]
    width_c = width // 2 + 1
    h_int_freq = np.zeros((n_f, n_s, height, width_c), dtype=np.complex128)

    for f, n in it.product(range(n_f), range(n_s)):
        h_int_freq[f, n] = ir2fr(h_int[f, n], shape=(height, width), real=True)

    return h_int_freq, h_int


# =============================================================================


def forward_f(filters, obj):
    """
    this function return the simulated data in frequency domain
    obj: the object
    filters : 4D-otf of the psf
    """
    return [np.sum(filt * urdft2(obj), axis=0) for filt in filters]


# =============================================================================


def forward(filters, obj):
    """
    this function return the simulated data in spatial domain
    obj: the object
    filters : 4D-otf of the psf
    """
    return [uirdft2(np.sum(filt * udft2(obj), axis=0)) for filt in filters]


# =============================================================================


def data_fidelity(H_freq, y_freq, in_param, x_freq):
    """ Compute norm(y-hx, 2) """
    H_x_freq = np.sum(H_freq * x_freq[np.newaxis, ...], axis=1, keepdims=True)
    J_crit = np.sum(np.abs(y_freq - H_x_freq) ** 2)

    # adding spatial regularization
    try:
        if in_param["mu_spat"] and np.any(in_param["D_spat"]):
            reg_spat = in_param["mu_spat"] * np.sum(
                norm_2_square(in_param["D_spat"] * x_freq_m)
                for x_freq_idx, x_freq_m in enumerate(x_freq)
            )
            J_crit += reg_spat
    except:
        pass

    # adding spectral regularization
    try:
        if in_param["mu_spec"] and np.any(in_param["D_spec"]):
            # reg_spec = in_param['mu_spec'] * np.sum(norm_2_square(in_param['D_spec'] * x_freq_m) \
            #                                         for x_freq_idx, x_freq_m in enumerate(x_freq))
            x = uirdft2(x_freq)
            x_freq_spec = np.fft.rfft(x, axis=0)  # dft sur l'axe spectral de l'objet
            reg_spec = in_param["mu_spec"] * norm_2_square(
                in_param["D_spec"][..., np.newaxis, np.newaxis] * x_freq_spec
            )
            J_crit += reg_spec
    except:
        pass

    # reg_spec = np.sum(mu * np.sum(np.abs(D_spat_f * obj_f[m])**2)
    # reg_spec = in_param['mu_spec'] * np.sum(np.abs((in_param['D_spec_freq']).reshape(-1,1,1) * x_freq)**2)

    return J_crit


# =============================================================================


def calc_second_term(H_freq, y_freq):
    """Compute b = -q in a frequency-domain, b_freq,
    psfs_freq: 4-D kernels , (filterIndex, ObjectIndex, 2-spatialIndex)
    y_freq : cube of observed data in frequency domain
    output : H^t y, cube of size [n_lam_obj, height, width]"""

    # compute H^t * y
    return np.sum(
        2 * np.swapaxes(np.conj(H_freq), 0, 1) * y_freq[np.newaxis, ...], axis=1
    )


# =============================================================================


def compute_hessian_x(H_freq, in_param, x_freq):
    """This function compute the product Qx according an analytic exppresions,
    x_freq : is a vector, each element of it represent is an 2D array of the
    object, H_freq: 4-D kernels , (filterIndex, ObjectIndex, 2-spatialIndex)
    mu:  vector contains regularization parameter of each object
    D : is an operatofr of the regularization parameter"""

    # (H^t*H)x
    sum_n = np.sum(H_freq * x_freq[np.newaxis, ...], axis=1, keepdims=True)
    HH_x_freq = np.sum(np.conj(H_freq) * sum_n, axis=0)
    Q_x = HH_x_freq

    # (D_{spat}^t*D_{spat})x
    try:
        DD_spat_x = (
            in_param["mu_spat"]
            * (np.abs(in_param["D_spat"][np.newaxis, ...]) ** 2)
            * x_freq
        )
        Q_x += DD_spat_x
    except:
        pass

    # (D_{spec}^t*D_{spec})x
    try:
        if in_param["mu_spec"]:
            x = uirdft2(x_freq)
            x_freq_spec = np.fft.rfft(x, axis=0)  # dft sur l'axe spectral de l'objet
            DD_spec_x = (
                in_param["mu_spec"]
                * (np.abs(in_param["D_spec"]) ** 2).reshape(-1, 1, 1)
                * x_freq_spec
            )
            DD_spec_x = urdft2(np.fft.irfft(DD_spec_x, axis=0))
            Q_x += DD_spec_x
    except:
        pass

    # I*x
    try:
        if in_param["tau_lambda"]:
            I_freq = urdft2(np.ones(in_param["H"], in_param["W"]))
            Q_x = in_param["tau_lambda"] * Q_x + np.sum(I_freq * x_freq, axis=0)
    except:
        pass

    return 2 * Q_x


def compute_hessian_spatio_spec_x(H_freq, in_param, x_freq):
    """This function compute the product Qx according an analytic exppresions,
    x_freq : is a vector, each element of it represent is an 2D array of the
    object, H_freq: 4-D kernels , (filterIndex, ObjectIndex, 2-spatialIndex)
    mu:  vector contains regularization parameter of each object
    D : is an operatofr of the regularization parameter"""

    # (H^t*H)x
    sum_n = np.sum(H_freq * x_freq[np.newaxis, ...], axis=1, keepdims=True)
    HH_x_freq = np.sum(np.conj(H_freq) * sum_n, axis=0)
    Q_x = HH_x_freq

    # (D_{spat}^t*D_{spat})x

    DD_spat_x = (
        in_param["mu_spat"]
        * (np.abs(in_param["D_spat"][np.newaxis, ...]) ** 2)
        * x_freq
    )
    Q_x += DD_spat_x

    # (D_{spec}^t*D_{spec})x
    x = uirdft2(x_freq)
    x_freq_spec = np.fft.rfft(x, axis=0)  # dft sur l'axe spectral de l'objet
    DD_spec_x = (
        in_param["mu_spec"]
        * (np.abs(in_param["D_spec"]) ** 2).reshape(-1, 1, 1)
        * x_freq_spec
    )
    DD_spec_x = urdft2(np.fft.irfft(DD_spec_x, axis=0))
    Q_x += DD_spec_x

    # I*x
    I_freq = urdft2(np.ones(in_param["H"], in_param["W"]))
    Q_x = in_param["tau_lambda"] * Q_x + np.sum(I_freq * x_freq, axis=0)

    return 2 * Q_x


# %% =============================================================================
# # quality measurement
# ==============================================================================


def norm_2_square(x):
    return np.sum(np.abs(x) ** 2)


def norm_2_residual(obj_hat_f, obj_f):
    if np.all(obj_f):
        return np.sum(np.abs(obj_f - obj_hat_f) ** 2)


def get_mse(vref, vcmp):
    """
    Compute Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      MSE between `vref` and `vcmp`
    """

    r = np.asarray(vref, dtype=np.float64).ravel()
    c = np.asarray(vcmp, dtype=np.float64).ravel()
    return np.mean(np.fabs(r - c) ** 2)


# =============================================================================


def isnr_func(orig, restored, noisy):
    """Improvement in signal-to-noise ratio (ISNR)
    An objective measure of the quality of the restored image,
    Improvement in Signal-to-Noise Ratio (ISNR) defined as:
    isnr= 10log10(|f-g|^2/|f-f^|^2); where f, g and f^ are:
    respectively the original, observed, and estimated images"""

    return 10 * np.log10(
        (np.abs(orig - noisy) ** 2).sum() / (np.abs(orig - restored) ** 2).sum()
    )


# =============================================================================


def bsnr_func(blurred, noise_std):
    """   Blurred Signal-to-Noise Ratio (BSNR) : Katsaggelos   """
    e = blurred - np.mean(blurred)
    bsnr = (np.abs(e) ** 2).mean() / (noise_std ** 2)
    bsnr_dB = 10 * np.log10(bsnr)
    return bsnr_dB


# =============================================================================


def get_bsnr(vblr, vnsy):
    """
    Compute Blurred Signal to Noise Ratio (BSNR) for a blurred and noisy
    image.

    Parameters
    ----------
    vblr : array_like
      Blurred noise free image
    vnsy : array_like
      Blurred image with additive noise

    Returns
    -------
    x : float
      BSNR of `vnsy` with respect to `vblr` and `vdeg`
    """

    blrvar = np.var(vblr, axis=(1, 2))
    nsevar = np.var(vnsy - vblr, axis=(1, 2))
    with np.errstate(divide="ignore"):
        rt = blrvar / nsevar
    return 10.0 * np.log10(rt)


def get_snr(sig_ref, sig_nsy=None, noise=None, noise_var=None, log=True):
    """Signal-to-Noise ratio (SNR): dB; Gonzalez book, Digital image processing
    snr = (sig_ref**2).sum() / (noise**2).sum()
    snr = (sig_ref**2).sum() / ((sig_ref-sig_nsy)**2).sum()
    snr = (sig_ref**2).mean() / var_noise # Only for zero-mean noise
    20 log10(||noiseless||_2/||noise||_2)
    """

    if noise_var or noise_var == 0:
        # Measure original signal power
        n_pix = np.prod(sig_ref.shape)
        P_signal = np.sum(sig_ref ** 2) / n_pix
        print("P_signal={0:f}".format(P_signal))
        # Measure noise power
        P_noise = noise_var
    else:
        # Measure original signal power
        P_signal = np.sum(sig_ref ** 2)

        # Measure noise power
        if np.any(noise):
            P_noise = np.sum(noise ** 2)
        elif np.any(sig_nsy):
            noise = sig_ref - sig_nsy
            P_noise = np.sum(noise ** 2)

    snr = P_signal / P_noise
    if log:
        return 10 * np.log10(snr)
    else:
        return snr


# =============================================================================


def get_psnr(vref, vcmp, rng=None):
    """
    Compute Peak Signal to Noise Ratio (PSNR) of two images. The PSNR
    calculation defaults to using the less common definition in terms
    of the actual range (i.e. max minus min) of the reference signal
    instead of the maximum possible range for the data type
    (i.e. :math:`2^b-1` for a :math:`b` bit representation).

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rng : None or int, optional (default None)
      Signal range, either the value to use (e.g. 255 for 8 bit samples) or
      None, in which case the actual range of the reference signal is used

    Returns
    -------
    x : float
      PSNR of `vcmp` with respect to `vref`
    """

    if rng is None:
        rng = vref.max() - vref.min()
    dv = (rng + 0.0) ** 2
    with np.errstate(divide="ignore"):
        rt = dv / get_mse(vref, vcmp)
    return 10.0 * np.log10(rt)


# =============================================================================


def get_isnr(vref, vdeg, vrst):
    """
    Compute Improvement Signal to Noise Ratio (ISNR) for reference,
    degraded, and restored images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vdeg : array_like
      Degraded image
    vrst : array_like
      Restored image

    Returns
    -------
    x : float
      ISNR of `vrst` with respect to `vref` and `vdeg`
    """

    msedeg = get_mse(vref, vdeg)
    mserst = get_mse(vref, vrst)
    with np.errstate(divide="ignore"):
        rt = msedeg / mserst
    return 10.0 * np.log10(rt)


# =============================================================================


def relative_error(A, B):
    """..."""
    return (np.sqrt(np.sum((A - B) ** 2)) / np.sqrt(np.sum((A) ** 2))) * 100


# =============================================================================


def snr_func(ref, test=None, noise=None, log=False):
    """Signal-to-Noise ratio (SNR): dB; Gonzalez book, Digital image processing
    snr = (ref**2).sum() / (noise**2).sum()
    snr = (ref**2).sum() / ((ref-test)**2).sum()
    snr = (ref**2).mean() / var_noise # Only for zero-mean noise"""
    if np.any(noise):
        signal = (ref ** 2).sum()
        noise = (noise ** 2).sum()
        snr = signal / noise
        if log:
            return 10 * np.log10(snr)
        else:
            return snr
    elif np.any(test):
        signal = (ref ** 2).sum()
        noise = ((ref - test) ** 2).sum()
        snr = signal / noise
        if log:
            return 10 * np.log10(snr)
        else:
            return snr
    else:
        print("Error: one argument missing ")


def mse_func(ref, test):
    """   Mean-Square-Error (MSE):    """
    error = ref - test
    return np.mean(np.abs(error) ** 2)


# =============================================================================


def psnr_func(ref, test):
    """ psnr(dB) """
    mse = mse_func(ref, test)
    return 10 * np.log10(np.max(ref) ** 2 / mse)


# %% =============================================================================


def AWGN(signal, std):
    """ add white gaussian noise"""

    # generate zero-mean gaussian noise
    seed(9001)
    m, n = signal.shape
    noise = std * randn(m, n)

    return signal + noise


def add_noise_to_data(y, std):
    return np.asarray([AWGN(signal=y[f], std=std) for f in range(y.shape[0])])


# =============================================================================


def snr_to_std(signal, snr_dB):
    """This function determine the standard deviation for a given
    data and SNR in dB"""

    avg_energy = np.mean((signal) ** 2)
    snr_linear = 10 ** (snr_dB / 10)

    if snr_linear != 0:
        var = avg_energy / snr_linear
    else:
        raise ValueError("please set a non-zero SNR")
    return np.sqrt(var)


# ==============================================================================
def normalizeX(obj, type="Total"):
    """
    normalize the image
    Type = 'Peak', 'Total', '01'
    """
    if type == "01":
        if (obj.max() - obj.min()) != 0:
            obj = (obj - obj.min()) / (obj.max() - obj.min())
        else:
            print("image full of zeros")
    if type == "Total":
        obj = obj / obj.sum()
    elif type == "Peak":
        obj = obj / obj.max()
    else:
        raise ValueError("must chose the correct argument")
    return obj


# =============================================================================
def next2pow(inp):
    if inp <= 0:
        print("Error: input must be positive!\n")
        result = -1
    else:
        index = 0
        while (2 ** index) < inp:
            index = index + 1
        result = 2 ** index
    return result


# =============================================================================
def addGaussianNoise(image_original, noisePercent=None, std=None):
    height, width = image_original.shape

    if noisePercent:
        std = (image_original.max() - image_original.min()) * noisePercent
    elif std:
        std = std
    else:
        std = 1
    noise = std * randn(height, width)
    image_noisy = image_original + noise
    return image_noisy, std


# =============================================================================
def bsnr2std(blurred, bsnr):
    """ bsnr"""
    NORM_BLURRED = (np.abs(blurred - blurred.mean()) ** 2).mean()
    std = np.sqrt(NORM_BLURRED / (10 ** (bsnr / 10)))
    return std


# %% =============================================================================
def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


# %% =============================================================================
def imresample(image, source_pscale, target_pscale, interp_order=1):
    """
    Resample data array from one pixel scale to another
    The resampling ensures the parity of the image is conserved
    to preserve the centering.
    Parameters
    ----------
    image : `numpy.ndarray`
        Input data array
    source_pscale : float
        Pixel scale of ``image`` in arcseconds
    target_pscale : float
        Pixel scale of output array in arcseconds
    interp_order : int, optional
        Spline interpolation order [0, 5] (default 1: linear)
    Returns
    -------
    output : `numpy.ndarray`
        Resampled data array
    """
    old_size = image.shape[0]
    new_size_raw = old_size * source_pscale / target_pscale
    new_size = int(np.ceil(new_size_raw))

    if new_size > 10000:
        raise MemoryError(
            "The resampling will yield a too large image. "
            "Please resize the input PSF image."
        )

    # Chech for parity
    if (old_size - new_size) % 2 == 1:
        new_size += 1

    ratio = new_size / old_size
    #    print(ratio**2)    / ratio**2
    return zoom(image, ratio, order=interp_order)


# =============================================================================
def solid_angle_in_steradian(psc_arcsec):
    arc_to_ster = 2.3504430539097885e-11
    return psc_arcsec ** 2 * arc_to_ster  # steradian


# =============================================================================
def read_object_from_fits(
    fname_cube,
    fname_lambda,
    psc1,
    psc2,
    lambda_cut=None,
    display_slice=None,
    folder=None,
    unit_conv=None,
    display=None,
):
    """
    MJy.sr^-1
    # wave_obj, obj_pad, obj_MJy, obj_phot, conv_coef = phd.read_object_from_fits(fname_cube,
    # fname_lambda, psc1, psc2, lambda_cut=[pce_miri[0,0], pce_miri[0,-1]], \
    #                          display_slice=100, dir_result=dir_result, unit_conv = True,
    display=True)
    ### compute the total flux of the object
    # flux_obj_pad = phd.compute_flux(obj_pad, psc1)
    # flux_obj_psc = phd.compute_flux(obj_MJy, psc2)

    """
    ## open fits files
    hdulist_src = fits.open(fname_cube)
    hdulist_w = fits.open(fname_lambda)
    wave_obj = np.asarray(hdulist_w[0].data, dtype=np.float32)
    obj_raw = np.asarray(hdulist_src[0].data, dtype=np.float32)
    ## close fits files
    hdulist_src.close()
    hdulist_w.close()

    # crop the object spectrally to [4, 30] microns
    if lambda_cut:
        idx = np.where(
            np.logical_and((lambda_cut[0] < wave_obj), (wave_obj < lambda_cut[1]))
            == True
        )
        wave_obj = wave_obj[idx[0]]
        obj_raw = obj_raw[idx[0]]

    ## Padding the cube so it'll be square spatially
    n, m, k = obj_raw.shape
    obj_pad = np.lib.pad(obj_raw, ((0, 0), (0, max(m, k) - min(m, k)), (0, 0)), "edge")

    ## Change pixel scale (psc) of the obeject : 0.5" -> 0.11"
    obj_psc_MJy = np.array(
        [imresample(obj_pad[i], 0.5, 0.11, interp_order=1) for i in range(n)]
    )
    if unit_conv:
        conv_coef = flux_unit_conversion(wave_obj * 1e-6)
        obj_psc_phot = conv_coef.reshape(-1, 1, 1) * obj_psc_MJy

    if display:
        # display astrophysique object
        disp_obj_spat = {
            "xlabel": "100 arcsec",
            "ylabel": "100 arcsec",
            "title": "0.5 arcsec/pix," + "\n" + " MJy/sr",
            "vmin": None,
            "vmax": None,
            "fontsize": None,
        }

        disp_obj_spec = {
            "fname": "astrophysic_object.pdf",
            "xlabel": "Wavelength [microns]",
            "ylabel": r"phot.s$^{-1}$.microns.pix$^{-1}$",
            "title": "SED",
            "label": "20-100",
        }

        fig = plt.figure()
        plt.clf()
        fig.set_size_inches(17, 12, forward=True)
        plt.subplot(2, 3, 1)
        disp_imshow(obj_pad[100], disp_obj_spat)
        disp_obj_spat.update(
            {"title": "0.11 arcsec/pix," + "\n" + " MJy/sr", "xlabel": "", "ylabel": ""}
        )
        plt.subplot(2, 3, 2)
        disp_imshow(obj_psc_MJy[100], disp_obj_spat)
        disp_obj_spat.update(
            {"title": "0.11 arcsec/pix," + "\n" + r" phot.s$^{-1}$.microns.pix$^{-1}$"}
        )
        plt.subplot(2, 3, 3)
        disp_imshow(obj_psc_phot[100], disp_obj_spat)
        plt.subplot(2, 1, 2)
        disp_plot(wave_obj, obj_psc_phot[:, 20, 100], disp_obj_spec)
        disp_obj_spec.update({"label": "20-200"})
        plt.subplot(2, 1, 2)
        disp_plot(wave_obj, obj_psc_phot[:, 20, 200], disp_obj_spec)
        disp_obj_spec.update({"label": "20-300"})
        plt.subplot(2, 1, 2)
        disp_plot(wave_obj, obj_psc_phot[:, 20, 300], disp_obj_spec)
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder + "/objet_astrophysique.pdf", bbox_inches="tight")
    # plt.show()

    return wave_obj, obj_pad, obj_psc_MJy, obj_psc_phot, conv_coef


# =============================================================================
def error_flux(orig_flux, test_flux):
    residual = orig_flux - test_flux
    error = (residual / orig_flux) * 100
    return error


# =============================================================================
def compute_flux(obj, psc):
    solid_angle_1 = solid_angle_in_steradian(psc)
    return np.asarray(
        [(val.sum() * solid_angle_1) for idx, val in enumerate(obj)]
    )  # MJy


# =============================================================================
def flux_unit_conversion(lambda_0, A_tel_m2=None, pixel_arcsec=None):
    """compute coef of conversion
    Input unit: MJy.sr^-1
    Output unit : photon.s^-1.pixel^-1.microns^-1
    Example: DOI : 10.1086/682258 / Glasse 2015
    flux1 = phd.flux_unit_conversion(0.4, 5e-6, 25.032, 0.11)
    flux2 = phd.flux_unit_conversion(188, 20e-6, 25.032, 0.11)
    phd.flux_unit_conversion(np.array([5,20])*1e-6, 25.03, 0.11) * np.array([0.4, 188])
    1 arcsec = pi/(180*3600) rad and sterad = rad^2
    """
    #    h = 6.626e-34 #joule·s
    A_tel_m2 = 25.03 if A_tel_m2 is None else A_tel_m2  # m^2
    pixel_arcsec = 0.11 if pixel_arcsec is None else pixel_arcsec  # arcsec

    c = 2.998e8  # m/s
    hc = 1.99e-25  # joules m
    arc_to_ster = 2.3504430539097885e-11
    E = hc / lambda_0
    sr_inv = arc_to_ster * pixel_arcsec ** 2
    hz_to_meter = c / lambda_0 ** 2  # s^-1m^-1 = Hz.m^-1
    merter_to_micron = 1e-6
    tau_tel = 0.88
    conv_coef = (
        1e6
        * 1e-26
        * E ** -1
        * tau_tel
        * A_tel_m2
        * sr_inv
        * hz_to_meter
        * merter_to_micron
    )
    #    print(conv_coef)
    #    flux_conv = conv_coef.reshape(-1, flux_MJy_sr.ndim) * flux_MJy_sr
    #    print(str(flux_MJy_sr)+' MJy.sr^-1 = ' + str(flux_conv)+'
    # photon.s^-1.pixel^-1.microns^-1')

    return conv_coef


# =============================================================================
def json_default(obj):
    """# write to file.json"""
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError


# =============================================================================
def precond(H_freq, obj_freq):
    """ precondionner M(x)=(H^t H)^-1 x"""
    inv_h_freq_abs_2 = 1 / np.sum(np.abs(H_freq) ** 2, axis=0)
    return inv_h_freq_abs_2 * obj_freq


# =============================================================================
def simul_data(pce_obj, psf_obj, wave_obj_int, obj, in_param):
    """ ... """

    time_tic = time.time()

    # Simulation of data using instrument model
    y = instrument_model(wave_obj_int, pce_obj, psf_obj, obj, in_param)

    np.save(in_param["file_inf_dB"], y)

    n_pix = np.prod(y.shape)
    P_signal = np.sum(y ** 2) / n_pix
    in_param["std"].append(np.sqrt(P_signal / 10 ** (in_param["snr"] / 10)))

    #    # measure SNR
    #    snr_dB = get_snr(y, noise_var=in_param['std'][-1] ** 2)
    #    in_param['snr_dB'] = snr_dB
    #    print(" SNR global = {0:2.2f} dB\t std={1:f} ".format(in_param['snr_dB'], in_param['std'][-1]))

    # Adding white gaussian noise (AWGN)
    y_noisy = add_noise_to_data(y, std=in_param["std"][-1])

    y_noisy_freq = urdft2(y_noisy)
    time_toc = time.time() - time_tic

    # compute BSNR
    in_param["bsnr"] = []
    #    n_f = y.shape[0]
    #    for f in range(n_f):
    #        in_param['bsnr'].append((get_bsnr(y[f], y_noisy[f])))

    return y_noisy, y_noisy_freq, time_toc, in_param


# =============================================================================
# Restoration
# =============================================================================


def restoration_broadband(y, in_param, psf_broadband):
    """..."""

    broad_param = {"max_iter": 200}

    time_start = time.time()

    mu_est = np.zeros(in_param["P"])
    x_rec = np.zeros_like(y)

    for f in range(in_param["P"]):
        x_rec[f], chains = udeconv(
            y[f], psf_broadband[f], reg=in_param["D_spat"], user_params=broad_param
        )
        mu_est[f] = np.mean(chains["prior"]) / np.mean(chains["noise"])

    time_udeconv = time.time() - time_start

    return x_rec, time_udeconv, broad_param


# =============================================================================
def reconstruction_multichannel_CG(H_freq, y_freq, Quad_param, x_hat_freq=None):
    """ Resolve a linear system Qx=b """
    #
    #    if Quad_param['n_s'] == 1 or Quad_param['N_conv'] == 1:
    #        print('call: inversion by diagonalization in fourier space...')
    #
    #        def calcCLSFilter(H, D, mu):
    #            """constrained least square"""
    #            denom = np.abs(H) ** 2 + mu * np.abs(D) ** 2
    #            num = np.conj(H)
    #            return num / denom
    #
    #        time_start = time.time()
    #        G_freq = calcCLSFilter(H_freq, Quad_param['D_spat'], Quad_param['mu_spat'])
    #        x_rec_freq = G_freq * np.conj(H_freq) * y_freq  # (H^t H + mu D^t D)^-1 H^t y
    #        x_rec = np.real(uirdft2(x_rec_freq))
    #        time_cls = time.time() - time_start
    #        return x_rec, time_cls
    #    else:
    # print('call: conjugated gradient algorithm...')
    cg_param = {
        "f crit": functools.partial(data_fidelity, H_freq, y_freq, Quad_param),
        "cg max iter": Quad_param["cg_iter"],
        "cg min iter": Quad_param["cg_iter"],
    }

    first_term_freq = functools.partial(compute_hessian_x, H_freq, Quad_param)
    second_term_freq = calc_second_term(H_freq, y_freq)

    time_start = time.time()
    x_init_freq = urdft2(
        np.zeros((Quad_param["n_s"], Quad_param["H"], Quad_param["W"]), dtype=float)
    )

    x_rec_freq, cg_info, status = conj_grad(
        first_term_freq,
        x_init_freq,
        second_term_freq,
        user_settings=cg_param,
        precond=None,
    )
    x_rec = np.real(uirdft2(x_rec_freq))
    time_cg = time.time() - time_start

    if Quad_param["VERBOSE"]:
        fig_crit = plt.figure()
        fig_crit.set_size_inches(8, 5)
        plt.plot(
            cg_info["crit_val"],
            lw=2,
            label=r"$\mu$={0:2.2e}".format(Quad_param["mu_spat"]),
        )
        plt.xlabel("Iteration")
        plt.ylabel(r"$\left\Vert y-H\hat{x}\right\Vert^2_2$")
        plt.legend(fontsize=18)
        plt.show()
    return x_rec, cg_info, time_cg


def reconstruction_multichannel_CG_l2l1(
    first_term_freq, second_term_freq, H_freq, y_freq, Quad_param, x_hat_freq=None
):
    """ Resolve a linear system Qx=b """

    cg_param = {
        "f crit": functools.partial(data_fidelity, H_freq, y_freq, Quad_param),
        "cg max iter": Quad_param["cg_iter"],
        "cg min iter": Quad_param["cg_iter"],
    }

    first_term_freq = functools.partial(compute_hessian_x, H_freq, Quad_param)
    #    second_term_freq = calc_second_term(H_freq, y_freq)

    time_start = time.time()
    x_init_freq = urdft2(
        np.zeros((Quad_param["n_s"], Quad_param["H"], Quad_param["W"]), dtype=float)
    )

    x_rec_freq, cg_info, status = conj_grad(
        first_term_freq,
        x_init_freq,
        second_term_freq,
        user_settings=cg_param,
        precond=None,
    )
    x_rec = np.real(uirdft2(x_rec_freq))
    time_cg = time.time() - time_start

    if Quad_param["VERBOSE"]:
        fig_crit = plt.figure()
        fig_crit.set_size_inches(8, 5)
        plt.plot(
            cg_info["crit_val"],
            lw=2,
            label=r"$\mu$={0:2.2e}".format(Quad_param["mu_spat"]),
        )
        plt.xlabel("Iteration")
        plt.ylabel(r"$\left\Vert y-H\hat{x}\right\Vert^2_2$")
        plt.legend(fontsize=18)
        plt.show()
    return x_rec, cg_info, time_cg


# =============================================================================
# def restoration_proposed_2(H_freq, y_freq, in_param, x_hat_freq):
#     """ Resolve a linear system Qx=b """
#
#     Q_x_freq = functools.partial(compute_hessian_x, H_freq, in_param)
#     b_freq = calc_second_term(H_freq, y_freq)  # rename to : compute_second_term
#
#     cg_param = {'f crit': functools.partial(criterion, H_freq, y_freq, in_param),
#                 'cg max iter': in_param['iter'],
#                 'cg min iter': in_param['iter'],
#                 'norm 2 residual': functools.partial(norm_2_residual, x_hat_freq == None)}
#
#     time_start = time.time()
#
#     # precond_p = functools.partial(precond, H_freq)
#     x_init_freq = urdft2(np.zeros((in_param['N_s'], in_param['H'], in_param['W']), dtype=float))
#     x_rec_freq, cg_info, status = conj_grad(Q_x_freq, x_init_freq, b_freq, user_settings=cg_param, precond=None)
#     x_rec_CG = np.real(uirdft2(x_rec_freq))
#
#     time_cg = time.time() - time_start
#
#     return x_rec_CG, cg_info, time_cg


# """ DATA """

##

# def load_iris_data():
#     """ Exemple du site : https://plot.ly/ipython-notebooks/principal-component-analysis/ """
#     http_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#     df = pd.read_csv(
#             filepath_or_buffer=http_link,
#             header=None,
#             sep=',')
#     df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']  # name columns
#     df.dropna(how="all", inplace=True)  # drops the empty line at file-end
#     df.tail(n=10)  # Returns last n rows
#
#     # split data table into data X and class labels y
#     n_features = 4
#     X = df.iloc[:, 0:n_features].values
#     y = df.iloc[:, n_features].values
#
#     return X, y  # plotting histograms

# =============================================================================


def load_3D_orig_obj(obj_orig_name):
    """..."""

    # obj_orig_name = input("Enter object name 'horsehead' or 'blackbody': ").lower()
    #    assert obj_orig_name == 'cube_horsehead' or obj_orig_name == 'cube_blackbody' \
    #           or obj_orig_name == 'linear' or obj_orig_name == 'cube_mire' or \
    #           obj_orig_name == 'cube_ced_201' or obj_orig_name == 'cube_oph_fil' or \
    #           obj_orig_name == 'cube_NGC_7023_E' or obj_orig_name == 'cube_NGC_7023_NW' or \
    #           obj_orig_name == 'cune_cameraman' or obj_orig_name == 'cube_synthetic', 'Please check the entry:'

    if obj_orig_name == "cube_horsehead":
        phi_true = np.load("data/cube/" + obj_orig_name + "_256.npy")
    else:
        phi_true = np.load("data/cube/" + obj_orig_name + ".npy")

    return phi_true


#    if obj_orig_name == 'horsehead':
#        # data_size = [int(x) for x in input("Enter the data size : ").split()]
#        data_size = 256
#        in_param.update({'data_size': data_size})
#        obj_phot = np.load('data/data_alain_abergel/obj_psc_phot.npy')
#        wave_obj = np.load('data/data_alain_abergel/wave_obj.npy')
#
#    elif obj_orig_name == 'blackbody':
#        data_size = 256
#        in_param.update({'data_size': data_size})
#        param = dict(fluxunit='photlam', waveunit='angstrom', srcType='blackbody', wave_ref_microns=1,
#                     temp_kelvin=10255, f_tot=1, fwhm=1, wave_mean=15, amplitude=1, pl_ind=2)
#        src = create_src_spectrum(param)
#        src.convert('microns')
#        src.convert('Jy')
#
#        wave_obj_int = np.linspace(wave_miri[0], wave_miri[-1], 1000)
#        umflux = src.sample(wave_obj_int)
#        Gauss = gaussian_2d((64, 64), 6)
#        obj_phot = 1e6 * create_cube_from_spectrum([1000, data_size, data_size], umflux, Gauss)
#
#    elif obj_orig_name == 'linear':
#        data_size = 64
#        in_param.update({'data_size': data_size})
#        kernel = gaussian_2d(shape=(data_size, data_size), sigma=3.0)
#        src_level = 505.0
#        src_slope = 39.6 * np.array([-1])
#        height = width = data_size
#        n_lam_obj = 9
#        wave_obj = wave_band = np.array([5, 6.3, 8.8, 11.6, 13.8, 16.5, 19, 23.5, 27.5])
#        obj_phot = create_cube_linear_source(obj_shape=[n_lam_obj, height, width],
#                                             wave_axis=wave_band,
#                                             kernel=kernel,
#                                             features=[src_slope, src_level])
#    else :

#    elif obj_orig_name == 'mire' or obj_orig_name == 'resolution_chart':
#        wave_obj = np.load('data/data_alain_abergel/wave_obj.npy')
#        obj_phot = np.load('data/orig_obj/mire_cube_256_256.npy')
#        in_param.update({'data_size': 256})
#
#    elif obj_orig_name == 'cameraman':
#        wave_obj = np.load('data/data_alain_abergel/wave_obj.npy')
#        obj_phot = np.load('data/orig_obj/cameraman_cube_256_256.npy')
#        in_param.update({'data_size': 256})
#
#    elif obj_orig_name == 'synthetic':
#        wave_obj = np.load('data/data_alain_abergel/wave_obj.npy')
#        obj_phot = np.load('data/orig_obj/synthetic_3spectre.npy')
#        in_param.update({'data_size': 256})
#
#    elif obj_orig_name == 'synthetic':
#        wave_obj = np.load('data/orig_obj/'+obj_orig_name+'.npy')
#        obj_phot = np.load('data/orig_obj/synthetic_3spectre.npy')
#        in_param.update({'data_size': 256})


# =============================================================================
def PCA(X, n_princomp, implementation, display=False, preprocessing="normalization"):
    """..."""

    # Standardizing the data
    if preprocessing == "normalization":
        x_stand = np.subtract(X, np.mean(X, axis=0))
    elif preprocessing == "standardizing":
        from sklearn.preprocessing import StandardScaler

        x_stand = StandardScaler().fit_transform(X)
    else:
        raise ValueError("Please check the preprocessing parameter.")
        x_stand = None

    if implementation == "eig_decomp":
        # 1 - Eigen decomposition - Computing Eigenvectors and Eigenvalues
        # eigendecomposition on the covariance matrix
        n_sampl, p_feat = x_stand.shape
        mean_vec = np.mean(x_stand, axis=0)
        cov_mat = np.cov(x_stand.T)  # (X_stand).T.dot((X_stand)) / (n_sampl - 1)
        if display:
            print("Covariance matrix \n%s" % cov_mat)

        # Apply Singular Value Decomposition (SVD)
        U, s, V = np.linalg.svd(x_stand.T)

        # 2 - Selecting Principal Components
        print("Eigenvalues in descending order:")
        eig_vals = s ** 2 / n_sampl  # or variance explained
        print(s ** 2 / n_sampl)

        # how many PC are we going to choose for our new feature subspace ?
        tot = eig_vals.sum()
        var_exp_ratio = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]

        # construction of the projection matrix
        matrix_w = U[:, 0:n_princomp]

        if display:
            print("Matrix W:\n", matrix_w)

        # 3 - Projection Onto the New Feature Space
        Y = x_stand.dot(matrix_w)
        return x_stand, Y, var_exp_ratio

    elif implementation == "sklearn":

        sk_pca = PCA(n_components=n_princomp)  # PCA object
        # sk_pca.fit(x_stand)
        return sk_pca, x_stand

    # elif implementation == "mlab":
    #     from matplotlib.mlab import PCA
    #     pca = PCA(X_stand) # PCA object
    #     pca.numrows, pca.numcols = X_stand.shape
    #     Y_sk = pca.project(X_stand)
    #     return X_stand, Y_sk, pca
    else:
        raise ValueError("Please check the entry: must be 'eig_decomp' or 'sklearn'")


# =============================================================================
def compute_norm_diff(x_true, x_rec, norm=2):
    """compute distance between x and x_rec"""
    x_diff = np.subtract(x_true, x_rec)
    if norm == 1 or norm == None:
        return (np.sum(np.abs(x_true - x_rec)) / np.sum(np.abs(x_true))) * 100
    elif norm == "inf":
        return (np.max(np.abs(x_true - x_rec)) / np.max(np.abs(x_true))) * 100
    elif norm == 2:
        return 100 * (
            np.sqrt(np.sum(np.power(np.abs(x_diff), 2)))
            / np.sqrt(np.sum(np.power(np.abs(x_true), 2)))
        )
    else:
        raise ValueError("Please enter the correct norm: 1 or 2 or 'inf'")


def compute_NMSE(x_true, x_rec):
    return np.sum((x_true - x_rec) ** 2) / np.sum((x_true) ** 2)


def compute_NMAE(x_true, x_rec):
    return np.sum(np.abs(x_true - x_rec)) / np.sum(np.abs(x_true))


def get_idx_of_x_from_vect(vect, x):
    """find the osition of x in the vect vect based on linear interpolation"""
    return np.ceil(np.interp(x, vect, range(len(vect)))).astype(int)


# =============================================================================
# Total variation
# =============================================================================


def forward_gradient(im):
    """
    Function to compute the forward gradient of the image I.
    Definition from: http://www.ipol.im/pub/art/2014/103/, p208
    :param im: numpy array [MxN], input image
    :return: numpy array [MxNx2], gradient of the input image, the first channel is the horizontal gradient, the second
    is the vertical gradient.
    """
    h, w = im.shape
    gradient = np.zeros((h, w, 2), im.dtype)  # Allocate gradient array
    # Horizontal direction
    gradient[:, :-1, 0] = im[:, 1:] - im[:, :-1]
    # Vertical direction
    gradient[:-1, :, 1] = im[1:, :] - im[:-1, :]

    return gradient


def backward_divergence(grad):
    """
    Function to compute the backward divergence.
    Definition in : http://www.ipol.im/pub/art/2014/103/, p208
    :param grad: numpy array [NxMx2], array with the same dimensions as the gradient of the image to denoise.
    :return: numpy array [NxM], backward divergence
    """

    h, w = grad.shape[:2]
    div = np.zeros((h, w), grad.dtype)  # Allocate divergence array

    # Vertical direction
    d_v = np.zeros((h, w), grad.dtype)
    d_v[0, :] = grad[0, :, 1]  # i=1
    d_v[1:-1, :] = grad[1:-1, :, 1] - grad[:-2, :, 1]  # 1<i<M
    d_v[-1, :] = -grad[-2:-1, :, 1].flatten()  # i=M

    # Horizontal direction
    d_h = np.zeros((h, w), grad.dtype)
    d_h[:, 0] = grad[:, 0, 0]  # j=1
    d_h[:, 1:-1] = grad[:, 1:-1, 0] - grad[:, :-2, 0]  # 1<j<N
    d_h[:, -1] = -grad[:, -2:-1, 0].flatten()  # j=N

    # Divergence
    div = d_h + d_v
    return div


def anorm(x):
    """Calculate L2 norm over the last array dimension"""
    return np.sqrt((x * x).sum(-1))


def calc_energy_TVL1(x, y, clambda):
    Ereg = anorm(nabla(x)).sum()
    Edata = clambda * np.abs(x - y).sum()
    return Ereg + Edata


# =============================================================================
# @nb.jit
def shrink_1d(X, F, threshold):
    """pixel-wise scalar srinking"""
    return X + np.clip(F - X, -threshold, threshold)


# =============================================================================
def reconstruction_tv(y, x_init, tv_param):
    """sources :
    http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    http://www.numerical-tours.com/matlab/optim_5_primal_dual/
    """
    # convergence parameters of primal dual algorithm
    L_square = 8.0  # 2.82
    tau = 0.05
    sigma = 1.0 / (L_square * tau)
    theta = 1.0
    J_crit = []

    if np.ndim(y) == 2:
        # ===========================================================================
        def nabla(Im):
            """
            Function to compute the forward gradient of the image I.
            Definition from: http://www.ipol.im/pub/art/2014/103/, p208
            :param im: numpy array [MxN], input image
            :return: numpy array [MxNx2], gradient of the input image,
            the first channel is the horizontal gradient, the second
            is the vertical gradient.
            https://github.com/louisenaud/primal-dual/blob/master/differential_operators.py
            """
            h, w = Im.shape
            Gradient = np.zeros((h, w, 2), Im.dtype)
            Gradient[:, :-1, 0] = Im[:, 1:] - Im[:, :-1]  # h: Im_(i,j+1)-Im_(i,j)
            Gradient[:-1, :, 1] = Im[1:, :] - Im[:-1, :]  # v: Im_(i+1,j)-Im_(i,j)
            return Gradient

        # ===========================================================================
        def nablaT(grad):
            """
            Function to compute the backward divergence.
            Definition in : http://www.ipol.im/pub/art/2014/103/, p208
            :param grad: numpy array [NxMx2], array with the same dimensions
            as the gradient of the image to denoise.
            :return: numpy array [NxM], backward divergence
            """
            h, w = grad.shape[:2]
            div = np.zeros((h, w), grad.dtype)  # Allocate divergence array

            # Vertical direction
            d_v = np.zeros((h, w), grad.dtype)
            d_v[0, :] = grad[0, :, 1]  # i=1
            d_v[1:-1, :] = grad[1:-1, :, 1] - grad[:-2, :, 1]  # 1<i<M
            d_v[-1, :] = -grad[-2:-1, :, 1].flatten()  # i=M

            # Horizontal direction
            d_h = np.zeros((h, w), grad.dtype)
            d_h[:, 0] = grad[:, 0, 0]  # j=1
            d_h[:, 1:-1] = grad[:, 1:-1, 0] - grad[:, :-2, 0]  # 1<j<N
            d_h[:, -1] = -grad[:, -2:-1, 0].flatten()  # j=N

            # Divergence
            div = d_h + d_v
            return div

        # ===========================================================================
        def project_nd(P, r):
            """perform a pixel-wise projection onto R-radius balls"""
            nP = np.maximum(1.0, anorm(P) / r)
            return P / nP[..., np.newaxis]

        # ===========================================================================
        def calc_energy_ROF(x, y, clambda, kernel=None):
            Ereg = anorm(nabla(x)).sum()

            try:
                # print('in')
                H_x = np.real(uirdft2(urdft2(x) * ir2fr(kernel, x.shape, real=True)))
            except:
                H_x = x

            Edata = 0.5 * clambda * np.sum((H_x - y) ** 2)
            return Ereg + Edata

        x_rec = x_init
        P = nabla(x_rec)

        if tv_param["app"] == "2d_deconv":  # model == "TV-L2"
            # print('deconvolution : TV-L2...')
            H_freq = ir2fr(tv_param["kernel"], y.shape, real=True)
            y_freq = urdft2(y)

            def prox_G(u, clambda, tau):
                num = tau * clambda * y_freq * np.conj(H_freq) + urdft2(u)
                den = tau * clambda * np.abs(H_freq) ** 2 + 1
                return uirdft2(num / den)

            calc_energy = lambda x, y, clambda: calc_energy_ROF(
                x, y, clambda, tv_param["kernel"]
            )

        elif tv_param["app"] == "2d_inpaint":  # model == "TV-L2"
            print("inpainting : TV-L2...")
            mask = tv_param["kernel"]
            prox_G = lambda u, clambda, tau: u * (1 - mask) + (
                (u + clambda * tau * y) / (1.0 + clambda * tau)
            ) * (mask)

        elif tv_param["app"] == "2d_denois":  # model == "TV-L2" or "TV-L1"
            print("denoising : " + tv_param["model"] + "...")
            if tv_param["model"] == "TV-L2":  # G_rof = \lambda/2 ||x-y||_2^2 "TV-L2"
                prox_G = lambda u, clambda, tau: (u + clambda * tau * y) / (
                    1.0 + clambda * tau
                )
                calc_energy = lambda x, y, clambda: calc_energy_ROF(x, y, clambda)

            elif tv_param["model"] == "TV-L1":  # G_rof = \lambda/2 ||x-y||_1
                prox_G = lambda u, clambda, tau: shrink_1d(u, y, clambda * tau)
                calc_energy = lambda x, y, clambda: calc_energy_TVL1(x, y, clambda)

            else:
                raise ValueError('please enter the correct model: "TV-L2" or "TV-L1"')

        else:
            raise ValueError(
                'please enter the correct application: "phd" or "deconv" or "inpaint" or "denois"'
            )

    elif np.ndim(y) == 3:
        # =============================================================================
        def nabla(I):
            n_s, h, w = I.shape
            Grad = np.zeros((n_s, h, w, 2), I.dtype)
            for i in range(n_s):
                Grad[i, :, :-1, 0] = I[i, :, 1:] - I[i, :, :-1]  # h
                Grad[i, :-1, :, 1] = I[i, 1:, :] - I[i, :-1, :]  # v
            return Grad

        # =============================================================================
        def nablaT(Grad):
            n, h, w = Grad.shape[:3]
            I = np.zeros((n, h, w), Grad.dtype)
            # note that we just reversed left and right sides
            # of each line to obtain the transposed operator
            for i in range(n):
                I[i, :, :-1] -= Grad[i, :, :-1, 0]
                I[i, :, 1:] += Grad[i, :, :-1, 0]
                I[i, :-1] -= Grad[i, :-1, :, 1]
                I[i, 1:] += Grad[i, :-1, :, 1]
            return I

        # =============================================================================
        def project_nd(P, r):
            """perform a pixel-wise projection onto R-radius balls"""
            n, h, w, n_grad = P.shape
            for i in range(n):
                nP = np.maximum(1.0, anorm(P[i]) / r)
                P[i] = P[i] / nP[..., np.newaxis]
            return P

        x_rec = np.zeros((tv_param["n_s"], tv_param["H"], tv_param["W"]))
        P = nabla(x_rec)

        H_freq = tv_param["kernel"]
        y_freq = urdft2(y)
        tv_param.update({"tau_lambda": tv_param["clambda"] * tau})

        # =============================================================================
        def prox_G(u_tild, clambda, tau):
            """   Proximqal operator of the function G()=||Hx-y||^2   """

            # compute u using conjugated gradient
            second_term_freq = urdft2(u_tild) + tau * clambda * calc_second_term(
                H_freq, y_freq
            )
            first_term_freq = functools.partial(compute_hessian_x, H_freq, tv_param)
            cg_param = {
                "f crit": functools.partial(data_fidelity, H_freq, y_freq),
                "cg max iter": tv_param["cg_iter"],
                "cg min iter": tv_param["cg_iter"],
            }

            x_0_freq = urdft2(
                np.zeros((tv_param["n_s"], tv_param["H"], tv_param["W"]), dtype=float)
            )
            time_start = time.time()
            u_freq, cg_info, status = conj_grad(
                first_term_freq, x_0_freq, second_term_freq, cg_param
            )
            time_cg = time.time() - time_start

            if tv_param["VERBOSE"] and (k % 10) == 0:
                print("time_CG = {0:2.2f} seconds".format(time_cg))
                fig_crit = plt.figure()
                fig_crit.set_size_inches(8, 10)
                plt.plot(cg_info["crit_val"], lw=2)
                plt.xlabel("Iteration")
                plt.ylabel(r"$||y-H\hat{x}||$")
                plt.close(fig_crit)
            return uirdft2(u_freq)

        # =============================================================================
        def calc_energy_ROF(x, y, clambda, kernel=None):
            Ereg = anorm(nabla(x)).sum()
            x_freq = urdft2(x)
            H_x = np.sum(H_freq * x_freq[np.newaxis, ...], axis=1, keepdims=True)
            Edata = 0.5 * clambda * np.sum((H_x - y) ** 2)
            return Ereg + Edata

        calc_energy = lambda x, y, clambda: calc_energy_ROF(x, y_freq, clambda, H_freq)

    else:
        raise ValueError("Only 2d or 3D data are accepted")

    for k in range(tv_param["niter"]):
        x_rec_freq = urdft2(x_rec)
        # %        J_crit.append(data_fidelity(H_freq, y_freq, x_rec_freq))

        P = project_nd(P + sigma * nabla(x_rec), 1.0)
        X1 = prox_G(x_rec - tau * nablaT(P), tv_param["clambda"], tau)
        x_rec = X1 + theta * (X1 - x_rec)

    return x_rec, J_crit


def chambolle_pock(P, P_t, data, Lambda, L, n_it, return_energy=True, verbose=False):
    """
    Chambolle-Pock algorithm for the minimization of the objective function
        ||P*x - d||_2^2 + Lambda*TV(x)

    P : projection operator (or H)
    PT : backprojection operator (or H_t)
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    """
    # Set parameter
    sigma = 1.0 / L
    tau = 1.0 / L
    theta = 1.0

    x = 0 * P_t(data)
    p = 0 * gradient(x)
    q = 0 * data
    x_tilde = 0 * x

    if return_energy:
        en = np.zeros(n_it)

    for k in range(0, n_it):

        # Update dual variables
        p = proj_l2(p + sigma * gradient(x_tilde), Lambda)
        q = (q + sigma * P(x_tilde) - sigma * data) / (1.0 + sigma)

        # Update primal variables
        x_old = x
        x = x + tau * div(p) - tau * P_t(q)  # eq 2 of Algo 3
        x_tilde = x + theta * (x - x_old)  # eq 3 of Algo 3

        # Calculate norms
        if return_energy:
            fidelity = 0.5 * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = 1.0 * fidelity + Lambda * tv
            en[k] = energy
            if verbose and k % 20 == 0:
                print(
                    "[%d] : energy %2.2e \t fidelity %2.2e \t TV %2.2e"
                    % (k, energy, fidelity, tv)
                )
    if return_energy:
        return en, x
    else:
        return x


def power_method(P, PT, data, n_it=10):
    """
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    """
    x = PT(data)
    for k in range(0, n_it):
        x = PT(P(x)) - div(gradient(x))
        s = np.sqrt(norm2sq(x))
        x /= s
    return np.sqrt(s)


def gradient(img):
    """
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    shape = [
        img.ndim,
    ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [
        0,
        slice(None, -1),
    ]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


def div(grad):
    """
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())


def norm1(mat):
    return np.sum(np.abs(mat))


def proj_l2(g, Lambda=1.0):
    """
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    """
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g ** 2, 0)) / Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res


# =============================================================================
# Reconstruct 3D spatio-spectral object
# =============================================================================
# @nb.jit
def x_to_phi(C_n, S_n):
    """..."""
    n_s, height, width = C_n.shape
    __, n_lam = S_n.shape
    phi_rec = np.zeros((n_lam, height, width))

    for i in range(n_s):
        phi_rec = (
            phi_rec + C_n[i][np.newaxis, ...] * S_n[i][..., np.newaxis, np.newaxis]
        )
    return phi_rec


def create_3D_object_from_2D(filename, spectre):
    import matplotlib.image as mpimg

    obj = np.array(mpimg.imread(filename), dtype=np.float32)
    im = (obj - obj.min()) / (obj.max() - obj.min())
    #    plt.figure()
    #    plt.imshow(im)

    if spectre == None:
        spectre = np.load("data/spectre_alain_1_pix.npy")
    phi = im[np.newaxis, ...] * spectre[:, np.newaxis, np.newaxis]
    # np.save('_cube_256_256.npy', phi)
    # gg = np.load('n_cube_256_256.npy')
    return phi


# ==================================================================================================
# hald quadratic: L2-L1
# ==================================================================================================

# @nb.jit(nopython=True, nogil=True)
def piecewise_func(A, B, cond, thresh):
    return A * (cond < thresh) + B * (cond >= thresh)


# ==================================================================================================


def restoration_HQ_GY(y, h, param):
    """..."""

    height, width = y.shape
    thresh = param["seuil"]
    alpha = param["alpha"]
    mu_ = param["mu"] / (2 * alpha)
    niter = param["niter"]
    x_rec_freq = urdft2(y)
    H_freq = ir2fr(h, shape=(height, width), real=True)
    y_freq = urdft2(y)

    if param["gradient"] == "separable":
        diff_kernel_row = (np.array([-1, 1]))[..., np.newaxis]
        diff_kernel_col = (np.array([-1, 1]))[np.newaxis, ...]

        # Fourier Space
        width_half = width // 2 + 1
        b_aux = np.zeros((2, height, width), dtype=np.float64)
        D_freq = np.zeros((2, height, width_half), dtype=np.complex128)
        D_freq[0] = ir2fr(diff_kernel_row, shape=(height, width), real=True)
        D_freq[1] = ir2fr(diff_kernel_col, shape=(height, width), real=True)

        InvHess = np.abs(H_freq) ** 2 + mu_ * (
            np.abs(D_freq[0]) ** 2 + np.abs(D_freq[1]) ** 2
        )
        InvHess = 1 / InvHess

        for k in range(niter):
            #  update b_row
            delta_x = np.real(uirdft2(D_freq[0] * x_rec_freq))
            A = delta_x * (1 - 2 * alpha)
            B = delta_x - 2 * alpha * thresh * np.sign(delta_x)
            b_aux[0] = piecewise_func(A, B, cond=np.abs(delta_x), thresh=thresh)

            #  update b_col
            delta_x = np.real(uirdft2(D_freq[1] * x_rec_freq))
            A = delta_x * (1 - 2 * alpha)
            B = delta_x - 2 * alpha * thresh * np.sign(delta_x)
            b_aux[1] = piecewise_func(A, B, cond=np.abs(delta_x), thresh=thresh)

            # update x
            b_freq = urdft2(b_aux)
            x_rec_freq = InvHess * (
                np.conj(H_freq) * y_freq
                + mu_
                * (np.conj(D_freq[0]) * b_freq[0] + np.conj(D_freq[1]) * b_freq[1])
            )

        x_rec = uirdft2(x_rec_freq)

        return [np.real(x_rec), b_aux]

    elif param["gradient"] == "joint":
        diff_kernel = np.array([[0, -1, 0], [-1, 4.0, -1], [0, -1, 0]], dtype=np.float)
        D_freq = ir2fr(diff_kernel, shape=(height, width), real=True)
        InvHess = np.abs(H_freq) ** 2 + mu_ * (np.abs(D_freq) ** 2)
        InvHess = 1 / InvHess

        for k in range(niter):
            #  update b
            d_x = np.real(uirdft2(D_freq * x_rec_freq))
            A = d_x * (1 - 2 * alpha)
            B = d_x - 2 * alpha * thresh * np.sign(d_x)
            b_aux = piecewise_func(A, B, cond=np.abs(d_x), thresh=thresh)

            # update x
            b_freq = urdft2(b_aux)
            x_rec_freq = InvHess * (
                np.conj(H_freq) * y_freq + mu_ * (np.conj(D_freq) * b_freq)
            )

        x_rec = uirdft2(x_rec_freq)

        return [np.real(x_rec), b_aux]


# ==================================================================================================
def first_term(H_freq, param, x_freq):
    """This function compute the product Qx according an analytic exppresions,
    x_freq : is a vector, each element of it represent is an 2D array of the
    object, H_freq: 4-D kernels , (filterIndex, ObjectIndex, 2-spatialIndex)
    mu:  vector contains regularization parameter of each object
    D : is an operatofr of the regularization parameter"""

    # (H^t*H)x
    sum_n = np.sum(H_freq * x_freq[np.newaxis, ...], axis=1, keepdims=True)
    HH_x_freq = np.sum(np.conj(H_freq) * sum_n, axis=0)

    # (D_{spat}^t*D_{spat})x
    mu_ = param["mu_spat"] / (2 * param["alpha"])
    DD_freq = np.abs(param["D_spat"][np.newaxis, ...]) ** 2
    DD_spat_x_freq = mu_ * DD_freq * x_freq

    return 2 * (HH_x_freq + DD_spat_x_freq)


# ==================================================================================================


def second_term(H_freq, y_freq, param, b_freq):
    """Compute b_aux = -q in a frequency-domain, b_freq,
    psfs_freq: 4-D kernels , (filterIndex, ObjectIndex, 2-spatialIndex)
    y_freq : cube of observed data in frequency domain
    output : H^t y, cube of size [n_lam_obj, height, width]"""

    prod_freq = np.conj(np.swapaxes(H_freq, 0, 1)) * y_freq[np.newaxis, ...]
    H_t_y_freq = np.sum(2 * prod_freq, axis=1)
    cg_param = {
        "f crit": functools.partial(data_fidelity, H_freq, y_freq, Quad_param),
        "cg max iter": Quad_param["cg_iter"],
        "cg min iter": Quad_param["cg_iter"],
    }

    first_term_freq = functools.partial(compute_hessian_x, H_freq, Quad_param)
    #    second_term_freq = calc_second_term(H_freq, y_freq)

    time_start = time.time()
    x_init_freq = urdft2(
        np.zeros((Quad_param["n_s"], Quad_param["H"], Quad_param["W"]), dtype=float)
    )

    x_rec_freq, cg_info, status = conj_grad(
        first_term_freq,
        x_init_freq,
        second_term_freq,
        user_settings=cg_param,
        precond=None,
    )
    x_rec = np.real(uirdft2(x_rec_freq))
    time_cg = time.time() - time_start
    mu_ = param["mu_spat"] / (2 * param["alpha"])
    D2_freq = mu_ * (np.conj(param["D_spat"])[np.newaxis, ...] * b_freq)

    return H_t_y_freq + D2_freq


def huber_value(threshold, obj):
    return np.sum(np.where(np.abs(obj) <= threshold, obj ** 2, np.abs(obj)))


def NQ_criterion(y_freq, x_rec_freq, H_freq, mu, thresh, D_freq):
    data_fid = norm_2_square(y_freq - compute_H_x(H_freq, x_rec_freq))

    delta_x_0 = np.real(uirdft2(D_freq[0][np.newaxis, ...] * x_rec_freq))
    A = delta_x_0 ** 2
    B = 2 * thresh * np.abs(delta_x_0) - thresh ** 2
    huber_value_0 = piecewise_func(A, B, cond=np.abs(delta_x_0), thresh=thresh)

    delta_x_1 = np.real(uirdft2(D_freq[1][np.newaxis, ...] * x_rec_freq))
    A = delta_x_1 ** 2
    B = 2 * thresh * np.abs(delta_x_1) - thresh ** 2
    huber_value_1 = piecewise_func(A, B, cond=np.abs(delta_x_1), thresh=thresh)

    return data_fid + mu * np.sum(huber_value_0 + huber_value_1)


# ==================================================================================================
def reconstruction_multichannel_HQ_GY(y_freq, H_freq, param):
    """..."""

    __, M, hei_freq, wid_freq = H_freq.shape
    x_rec_freq = np.zeros((M, hei_freq, wid_freq), dtype=np.complex128)

    niter = param["niter"]
    thresh = param["seuil"]
    alpha = param["alpha"]
    param["mu"] = param["mu"] / (2 * alpha)
    height = param["H"]
    width = param["W"]

    if param["gradient"] == "separable":
        diff_kernel_row = (np.array([-1, 1]))[..., np.newaxis]
        diff_kernel_col = (np.array([-1, 1]))[np.newaxis, ...]

        # Fourier Space
        width_half = width // 2 + 1
        b_aux = np.zeros((2, M, height, width), dtype=np.float64)
        D_freq = np.zeros((2, height, width_half), dtype=np.complex128)
        D_freq[0] = ir2fr(diff_kernel_row, shape=(height, width), real=True)
        D_freq[1] = ir2fr(diff_kernel_col, shape=(height, width), real=True)
        param["D_freq"] = D_freq
        param["crit"] = []
        x_rec = np.zeros((M, param["H"], param["W"]), dtype=np.float64)
        param.update({"stop_at_iter": niter})

        hess_freq = compute_hessian_l2l1(H_freq, param)
        inv_hess_freq = compute_inv_hessian(hess_freq)
        Hty = compute_H_t_y(H_freq, y_freq)

        for k in range(niter):
            #            print(k)
            #  update b
            delta_x = np.real(uirdft2(D_freq[0] * x_rec_freq))
            A = delta_x * (1 - 2 * alpha)
            B = delta_x - 2 * alpha * thresh * np.sign(delta_x)
            b_aux[0] = piecewise_func(A, B, cond=np.abs(delta_x), thresh=thresh)

            delta_x = np.real(uirdft2(D_freq[1] * x_rec_freq))
            A = delta_x * (1 - 2 * alpha)
            B = delta_x - 2 * alpha * thresh * np.sign(delta_x)
            b_aux[1] = piecewise_func(A, B, cond=np.abs(delta_x), thresh=thresh)

            param["b_aux"] = b_aux
            param["b_aux_freq"] = urdft2(b_aux)

            # update x
            x_rec_old = x_rec
            x_rec, param = reconstruction_multichannel_MDFT_l2l1(
                y_freq,
                H_freq,
                param,
                hess_freq=hess_freq,
                inv_hess_freq=inv_hess_freq,
                Hty=Hty,
            )

            param["stop"] = compute_norm_diff(x_rec, x_rec_old, norm=2)

            # evaluate the objective function
            x_rec_freq = urdft2(x_rec)
            param["crit"].append(
                NQ_criterion(y_freq, x_rec_freq, H_freq, param["mu"], thresh, D_freq)
            )

            if param["stop"] <= 1e-5:
                param.update({"stop_at_iter": k})
                return [np.real(x_rec), param, "ended by creterion"]

        return [np.real(x_rec), param, "ended by iteration"]

    elif param["gradient"] == "joint":
        diff_kernel = np.array([[0, -1, 0], [-1, 4.0, -1], [0, -1, 0]], dtype=np.float)
        D_freq = ir2fr(diff_kernel, shape=(height, width), real=True)
        param["D_freq"] = D_freq

        for k in range(niter):  # iteration on alternating minimization

            # update of b : separable problem
            delta = np.real(uirdft2(D_freq * x_rec_freq))
            A = delta * (1 - 2 * alpha)
            B = delta - 2 * alpha * thresh * np.sign(delta)
            b_aux = piecewise_func(A, B, cond=np.abs(delta), thresh=thresh)
            param["b_aux"] = b_aux
            param["b_aux_freq"] = urdft2(b_aux)

            # update of x : quadratic problem
            x_rec, param = reconstruction_multichannel_MDFT_l2l1(y_freq, H_freq, param)
            x_rec_freq = urdft2(x_rec)

    return [np.real(x_rec), param]


# ==================================================================================================
# ==================================================================================================

# @nb.jit
def compute_hessian(H_freq, param):
    """compute hessian matrix"""
    P, M, hei_freq, wid_freq = H_freq.shape
    hess_freq = np.zeros((M, M, hei_freq, wid_freq), dtype=np.complex128)
    H_t_H_freq = compute_H_t_H(H_freq)

    for i, j in it.product(range(M), range(M)):
        hess_freq[i, j] = H_t_H_freq[
            i, j
        ]  # np.sum(np.conj(H_freq[:, i]) * H_freq[:, j], axis=0)

        if i == j and param["reg"] == "spatio_spectral":
            mu_spat = param["mu_spat"]
            mu_spec = param["mu_spec"]
            D_spat = param["D_spat"]
            hess_freq[i, j] = (
                hess_freq[i, j] + mu_spat * np.abs(D_spat ** 2) + 2 * mu_spec
            )

        if (
            (i == j + 1)
            or (i + 1 == j)
            or (i == M - 1 and j == 0)
            or (i == 0 and j == M - 1)
        ) and param["reg"] == "spatio_spectral":
            hess_freq[i, j] = hess_freq[i, j] - mu_spec

        if i == j:
            if param["reg"] == "quad":
                D_freq = param["D_freq"]
                mu = param["mu"]
                hess_freq[i, j] = hess_freq[i, j] + mu * np.abs(D_freq ** 2)

            elif param["reg"] == "l2l1" and param["gradient"] == "joint":
                D_freq = param["D_freq"]
                mu_ = param["mu"] / (2 * param["alpha"])
                D_t_D_freq = np.abs(D_freq ** 2)
                hess_freq[i, j] = hess_freq[i, j] + mu_ * D_t_D_freq

            elif param["reg"] == "l2l1" and param["gradient"] == "separable":
                mu_ = param["mu"] / (2 * param["alpha"])
                D_t_D_freq = np.abs(D_freq[0]) ** 2 + np.abs(D_freq[1]) ** 2
                hess_freq[i, j] = hess_freq[i, j] + mu_ * D_t_D_freq
    return hess_freq


# ==================================================================================================
# @nb.jit
def compute_hessian_spatio_spec(H_freq, param):
    """compute hessian matrix"""
    P, M, hei_freq, wid_freq = H_freq.shape
    hess_freq = np.zeros((M, M, hei_freq, wid_freq), dtype=np.complex128)
    H_t_H_freq = compute_H_t_H(H_freq)

    for i, j in it.product(range(M), range(M)):
        hess_freq[i, j] = H_t_H_freq[
            i, j
        ]  # np.sum(np.conj(H_freq[:, i]) * H_freq[:, j], axis=0)

        if i == j and param["reg"] == "spatio_spectral":
            mu_spat = param["mu_spat"]
            mu_spec = param["mu_spec"]
            D_spat = param["D_spat"]
            hess_freq[i, j] = (
                hess_freq[i, j] + mu_spat * np.abs(D_spat ** 2) + 2 * mu_spec
            )

        if (
            (i == j + 1)
            or (i + 1 == j)
            or (i == M - 1 and j == 0)
            or (i == 0 and j == M - 1)
        ) and param["reg"] == "spatio_spectral":
            hess_freq[i, j] = hess_freq[i, j] - mu_spec

    return hess_freq


# ==================================================================================================

# @nb.jit
def compute_hessian_l2l1(H_freq, param):
    """compute hessian matrix"""
    P, M, hei_freq, wid_freq = H_freq.shape
    hess_freq = np.zeros((M, M, hei_freq, wid_freq), dtype=np.complex128)

    H_t_H_freq = compute_H_t_H(H_freq)

    mu = param["mu"]
    D_freq = param["D_freq"]

    for i, j in it.product(range(M), range(M)):
        hess_freq[i, j] = H_t_H_freq[i, j]

        if i == j:
            if param["reg"] == "quad":
                hess_freq[i, j] = hess_freq[i, j] + mu * np.abs(D_freq ** 2)

            if param["reg"] == "l2l1" and param["gradient"] == "joint":
                mu_ = mu / (2 * param["alpha"])
                D_t_D_freq = np.abs(D_freq ** 2)
                hess_freq[i, j] = hess_freq[i, j] + mu_ * D_t_D_freq

            elif param["reg"] == "l2l1" and param["gradient"] == "separable":
                mu_ = mu / (2 * param["alpha"])
                D_t_D_freq = np.abs(D_freq[0]) ** 2 + np.abs(D_freq[1]) ** 2
                hess_freq[i, j] = hess_freq[i, j] + mu_ * D_t_D_freq
    return hess_freq


# ==================================================================================================
# @nb.jit
def compute_second_term(H_freq, y_freq):
    """..."""
    prod_freq = np.conj(np.swapaxes(H_freq, 0, 1)) * y_freq[np.newaxis, ...]
    return np.sum(prod_freq, axis=1)


# ==================================================================================================
# @nb.jit
def compute_second_term_perturbation(H_freq, y_freq, param):
    """..."""

    # perturbation
    gamma_n = param["gamma_n"]
    gamma_s = param["gamma_s"]
    gamma_lam = param["gamma_lam"]
    prtb_freq = param["perturb"]
    D_spat = param["D_spat"]

    # H^t * \eta_1
    prod_freq = np.conj(np.swapaxes(H_freq, 0, 1)) * prtb_freq[np.newaxis, ...]
    H_t_y_freq = gamma_n * np.sum(prod_freq, axis=1)

    # D_s^t * \eta_1
    D_s_p_freq = gamma_s * np.conj(D_spat) * prtb_freq[np.newaxis, ...]

    # D_{lam}^t * \eta_3
    D_s_p = gamma_lam * np.conj(D_lam) * prtb_freq[np.newaxis, ...]

    return H_t_y_freq


# ==================================================================================================

# @nb.jit
def compute_L_t_b(param):
    """..."""
    mu_ = param["mu"] / (2 * param["alpha"])
    D_freq = param["D_freq"]
    b_aux_freq = param["b_aux_freq"]

    if param["gradient"] == "joint":
        return mu_ * (np.conj(D_freq)[np.newaxis, ...] * b_aux_freq)
    if param["gradient"] == "separable":
        return mu_ * (
            np.conj(D_freq[0]) * b_aux_freq[0] + np.conj(D_freq[1]) * b_aux_freq[1]
        )


# ==================================================================================================
# @nb.jit
def compute_inv_hessian(hess_freq):
    """optimiser par françois"""
    inv_hess_freq = np.zeros_like(hess_freq)

    for h, w in it.product(range(hess_freq.shape[2]), range(hess_freq.shape[3])):
        inv_hess_freq[:, :, h, w] = np.linalg.inv(hess_freq[:, :, h, w])

    # @nb.jit
    # def compute_inv_hessian(hess_freq):
    #     """..."""
    #     M, M, hei_freq, wid_freq = hess_freq.shape
    #     D = np.zeros((hei_freq * wid_freq, M, M), dtype=np.complex128)
    #     B = np.zeros_like(D)
    #     inv_hess_freq = np.zeros_like(hess_freq)
    #
    #     for k in range(hei_freq * wid_freq):
    #         for i in range(M):
    #             for j in range(M):
    #                 D[k, i, j] = hess_freq[i, j].flatten()[k]
    #         B[k] = np.linalg.inv(D[k])
    #
    #     for i in range(M):
    #         for j in range(M):
    #             inv_hess_freq[i, j] = B[:, i, j].reshape(hei_freq, wid_freq)
    #
    #     return inv_hess_freq

    return inv_hess_freq


# ==================================================================================================

# ==================================================================================================
# @nb.jit
def reconstruction_multichannel_DFT_spatio_spec(y_freq, H_freq, param):
    """multichannel regularized least squares"""

    time_start = time.time()
    hess_freq = compute_hessian_spatio_spec(H_freq, param)
    inv_hess_freq = compute_inv_hessian(hess_freq)
    x_rec_freq = np.sum(
        inv_hess_freq * compute_H_t_y(H_freq, y_freq)[np.newaxis, ...], axis=1
    )
    x_rec = np.real(uirdft2(x_rec_freq))
    param["time"] = time.time() - time_start

    return x_rec, param


# ==================================================================================================


def reconstruction_multichannel_MDFT_l2l1(
    y_freq, H_freq, param, hess_freq=None, inv_hess_freq=None, Hty=None
):
    """multichannel regularized least squares"""

    time_start = time.time()
    hess_freq = (
        hess_freq if hess_freq is not None else compute_hessian_l2l1(H_freq, param)
    )
    inv_hess_freq = (
        inv_hess_freq if inv_hess_freq is not None else compute_inv_hessian(hess_freq)
    )
    H_t_y_L_t_b_freq = Hty if Hty is not None else compute_H_t_y(H_freq, y_freq)
    H_t_y_L_t_b_freq = Hty + compute_L_t_b(param) if param["reg"] == "l2l1" else 0

    if param["method"] == "fft":
        x_rec_freq = np.sum(inv_hess_freq * H_t_y_L_t_b_freq[np.newaxis, ...], axis=1)
        x_rec = np.real(uirdft2(x_rec_freq))
        param["time"] = time.time() - time_start

    elif param["method"] == "cg":
        first_term_freq = functools.partial(compute_H_x, hess_freq)
        second_term_freq = H_t_y_L_t_b_freq

        cg_param = {
            "f crit": functools.partial(data_fidelity, H_freq, y_freq, param),
            "cg max iter": param["cg_iter"],
            "cg min iter": param["cg_iter"],
        }

        x_init_freq = urdft2(
            np.zeros((param["n_s"], param["H"], param["W"]), dtype=float)
        )

        x_rec_freq, cg_info, status = conj_grad(
            first_term_freq,
            x_init_freq,
            second_term_freq,
            user_settings=cg_param,
            precond=None,
        )
        x_rec = np.real(uirdft2(x_rec_freq))
        param["time"] = time.time() - time_start

    return x_rec, param


def reconstruction_multichannel_MDFT_l2l1_old(y_freq, H_freq, param):
    """multichannel regularized least squares"""

    time_start = time.time()
    hess_freq = compute_hessian_l2l1(H_freq, param)
    inv_hess_freq = compute_inv_hessian(hess_freq)
    H_t_y_L_t_b_freq = compute_H_t_y(H_freq, y_freq) + (
        compute_L_t_b(param) if param["reg"] == "l2l1" else 0
    )

    if param["method"] == "fft":
        x_rec_freq = np.sum(inv_hess_freq * H_t_y_L_t_b_freq[np.newaxis, ...], axis=1)
        x_rec = np.real(uirdft2(x_rec_freq))
        param["time"] = time.time() - time_start

    elif param["method"] == "cg":
        first_term_freq = functools.partial(compute_H_x, hess_freq)
        second_term_freq = H_t_y_L_t_b_freq

        cg_param = {
            "f crit": functools.partial(data_fidelity, H_freq, y_freq, param),
            "cg max iter": param["cg_iter"],
            "cg min iter": param["cg_iter"],
        }

        x_init_freq = urdft2(
            np.zeros((param["n_s"], param["H"], param["W"]), dtype=float)
        )

        x_rec_freq, cg_info, status = conj_grad(
            first_term_freq,
            x_init_freq,
            second_term_freq,
            user_settings=cg_param,
            precond=None,
        )
        x_rec = np.real(uirdft2(x_rec_freq))
        param["time"] = time.time() - time_start

    return x_rec, param


# ==================================================================================================
# ==================================================================================================
#               unsupervised spatio spectral multichannel reconstruction
# ==================================================================================================


# ==================================================================================================
# @nb.jit
def compute_H_x(H_freq, x_freq):
    """ compute H_freq * x_freq"""
    return np.sum(H_freq * x_freq[np.newaxis, ...], axis=1)


# ==================================================================================================
# ==================================================================================================
# @nb.jit
def D_lam_t_y(x):
    """..."""
    return np.roll(x, 1, axis=0) - x


# ==================================================================================================
# ==================================================================================================
# @nb.jit


def D_lam_x(x):
    """..."""
    return np.roll(x, -1, axis=0) - x


# ==================================================================================================

# @nb.jit
def compute_H_t_y(H_freq, y_freq):
    """ compute H_freq^t * y_freq"""
    return np.sum(np.swapaxes(np.conj(H_freq), 0, 1) * y_freq[np.newaxis, ...], axis=1)


# ==================================================================================================
# ==================================================================================================
# @nb.jit
def compute_H_t_H(H_freq):
    """ compute H_freq^t * H_freq"""
    P, M, hei_freq, wid_freq = H_freq.shape
    H_t_H = np.zeros(shape=(M, M, hei_freq, wid_freq), dtype=np.complex128)

    for i, j in it.product(range(M), range(M)):
        H_t_H[i, j] = np.sum(np.conj(H_freq[:, i]) * H_freq[:, j], axis=0)

    return H_t_H


# ==================================================================================================
# @nb.jit
def compute_hessian_spatio_spec_Gibbs_PO(H_freq, param):
    """compute hessian matrix"""

    P, M, hei_freq, wid_freq = H_freq.shape
    hess_freq = np.zeros((M, M, hei_freq, wid_freq), dtype=np.complex128)

    H_t_H_freq = compute_H_t_H(H_freq)

    Gamma_n = param["chain_gamma_n"][-1]
    Gamma_s = param["chain_gamma_s"][-1]
    Gamma_lam = param["chain_gamma_lam"][-1]

    D_spat_freq = param["D_spat"]
    aD_spat_freq = np.abs(D_spat_freq) ** 2

    for i, j in it.product(range(M), range(M)):

        hess_freq[i, j] = Gamma_n * H_t_H_freq[i, j]

        if i == j:
            hess_freq[i, j] = hess_freq[i, j] + Gamma_s * aD_spat_freq + 2 * Gamma_lam

        if (
            (i == j + 1)
            or (i + 1 == j)
            or (i == M - 1 and j == 0)
            or (i == 0 and j == M - 1)
        ):
            hess_freq[i, j] = hess_freq[i, j] - Gamma_lam

    return hess_freq


# ==================================================================================================
# @nb.jit
def compute_second_term_PO(H_freq, pert1_freq, pert2_freq, pert3_freq, param):
    """..."""

    Gamma_n = param["chain_gamma_n"][-1]
    Gamma_s = param["chain_gamma_s"][-1]
    Gamma_lam = param["chain_gamma_lam"][-1]
    D_spat_freq = param["D_spat"]

    return (
        Gamma_n * compute_H_t_y(H_freq, pert1_freq)
        + Gamma_s * np.conj(D_spat_freq)[np.newaxis, ...] * pert2_freq
        + Gamma_lam * D_lam_t_y(pert3_freq)
    )


# ==================================================================================================

# @nb.jit
def reconstruction_multichannel_MCMC(y_freq, H_freq, param):
    """multichannel regularized least squares"""
    time_start = time.time()

    # initializing parameter
    nbiter = param["nbiter"]
    nburn = nbiter / 10
    param.update({"chain_gamma_n": [1], "chain_gamma_s": [1], "chain_gamma_lam": [1]})
    P, M, hei_freq, wid_freq = H_freq.shape
    x_init = np.zeros(shape=(M, param["H"], param["W"]))
    x_freq = urdft2(x_init)
    x_freq_sum = np.zeros_like(x_freq)
    D_spat_freq = param["D_spat"]

    plt.figure(figsize=cm2inch(15, 5))
    # # PO
    # m_1 = m_2 = m_3 = np.zeros_like(x_freq)
    # R_1 = R_2 = R_3 = np.ones_like(H_freq)

    # Gibbs algorithm: sampling of conditional posteriors
    for i in range(nbiter):
        print("iteration = {0:d}".format(i))
        # etape d) estimation de l'image en fourier : loi gaussienne
        Q = compute_hessian_spatio_spec_Gibbs_PO(H_freq, param)
        R_x_post_freq = compute_inv_hessian(Q)
        m_x_post_freq = param["chain_gamma_n"][-1] * compute_H_x(
            R_x_post_freq, compute_H_t_y(H_freq, y_freq)
        )

        def RNDGauss(m_x_freq, R_x_freq):
            __, __, hei_freq, wid_freq = R_x_freq
            BoutGauss = npr.randn(M, hei_freq, wid_freq) + 1j * npr.randn(
                M, hei_freq, wid_freq
            )
            BoutGauss = urdft2(np.real(uirdft2(BoutGauss)))
            np.sqrt(0.5) * (
                np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)
            )
            return m_x_freq + compute_H_x(np.sqrt(R_x_freq), BoutGauss)

        x_freq = RNDGauss(m_x_post_freq, R_x_post_freq)

        # # Projection-Ootimization algorithm
        #
        # # Step P: ================== A revoir
        # pert1_freq = RNDGaussian(m_1, Gamma_n[i] * R_1)  # a revoir
        # pert2_freq = RNDGaussian(m_2, Gamma_s[i] * R_2)
        # pert3_freq = RNDGaussian(m_3, Gamma_lam[i] * R_3)
        #
        # # Step O:
        # sec_term = compute_second_term_PO(H_freq, pert1_freq, pert2_freq, pert3_freq, param)
        # x_freq = np.sum(R_x_post_freq * sec_term[np.newaxis, ...], axis=1)

        if i > nburn:
            x_freq_sum += x_freq

        #  etape a) estimation de gamma n: loi gamma
        beta = 2 / norm_2_square(y_freq - compute_H_x(H_freq, x_freq))
        alpha = 2 * y_freq.size / 2
        param["chain_gamma_n"].append(npr.gamma(shape=alpha, scale=beta))

        # etape b) estimation de gamma s: loi gamma
        beta = 2 / norm_2_square(D_spat_freq[np.newaxis, ...] * x_freq)
        alpha = (
            2 * x_freq.size - 1
        ) / 2  # prod(x_freq.shape)/2  M * hei_freq * wid_freq
        param["chain_gamma_s"].append(npr.gamma(shape=alpha, scale=beta))

        # etape c) estimation de gamma lam: loi gamma
        beta = 2 / norm_2_square(D_lam_x(x_freq))
        alpha = (2 * x_freq.size - 1) / 2
        param["chain_gamma_lam"].append(npr.gamma(shape=alpha, scale=beta))

        # interactive display
        plt.clf()
        plt.subplot(131), plt.plot(param["chain_gamma_n"], label="gn")
        plt.title("$\gamma_n$"), plt.xlabel("iterations")
        plt.subplot(132), plt.plot(param["chain_gamma_s"], label="gs")
        plt.title("$\gamma_s$"), plt.xlabel("iterations")
        plt.subplot(133), plt.plot(param["chain_gamma_lam"], label="glam")
        plt.title("$\gamma_{\lambda}$"), plt.xlabel("iterations")
        plt.pause(0.05)

    # problem solution : EAP
    x_freq_mean = x_freq_sum / (nbiter - nburn)
    x_rec = np.real(uirdft2(x_freq_mean))

    param["mu_spat_est"] = np.mean(param["chain_gamma_s"]) / np.mean(
        param["chain_gamma_n"]
    )
    param["mu_spec_est"] = np.mean(param["chain_gamma_lam"]) / np.mean(
        param["chain_gamma_n"]
    )
    param["time"] = time.time() - time_start

    return x_rec, param


def errorPerWavelength(phi_orig, phi_rec):
    phi_diff = np.subtract(phi_orig, phi_rec)
    num = np.sqrt(np.sum(np.power(phi_diff, 2), axis=(1, 2)))
    den = np.sqrt(np.sum(np.power(phi_orig, 2), axis=(1, 2)))
    error = np.divide(num, den)
    return 100 * np.nan_to_num(error, 0)


def errorPerSpatial(phi_orig, phi_rec):
    phi_diff = np.subtract(phi_orig, phi_rec)
    num = np.sqrt(np.sum(np.power(phi_diff, 2), axis=(0)))
    den = np.sqrt(np.sum(np.power(phi_orig, 2), axis=(0)))
    error = np.divide(num, den)
    return 100 * np.nan_to_num(error, 0)


# ==================================================================================================
