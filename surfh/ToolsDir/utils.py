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


def make_mask_FoV(cube, tol=10):
    mask = np.zeros(cube.shape[1:])
    cube[cube<tol] = 0
    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            if np.any(cube[:,i,j]):
                mask[i,j] = 1
    return mask

def apply_mask_FoV(mask, cube):
    masked = mask[np.newaxis,...] * cube
    masked[np.where(masked == 0)] = np.NaN
    return masked