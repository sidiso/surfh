# Copyright (C) 2022 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>

import numpy as np
import udft


def gaussian_psf(wavel_axis, step, D=6.5):
    x_axis = np.linspace(-30, 30, 40).reshape((1, -1))
    y_axis = x_axis.reshape((-1, 1))
    psf = np.empty((len(wavel_axis), len(y_axis), len(y_axis)))

    for w_idx, wavel in enumerate(wavel_axis):
        fwhm_arcsec = (wavel * 1e-6 / D) * 206265  # from rad to arcsec
        sigma = fwhm_arcsec / (step * 2.354)  # sigma in pixels
        psf[w_idx] = np.exp(-(x_axis**2 + y_axis**2) / (2 * sigma**2))

    return psf / np.sum(psf, axis=(1, 2), keepdims=True)


def otf(psf, shape, components):
    otf = udft.ir2fr(
        psf[np.newaxis, ...] * components[:, :, np.newaxis, np.newaxis], shape
    )
    return otf
