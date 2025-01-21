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

try:
    import skimage.measure as measure

    skimage_available = True
except ImportError:
    skimage_available = False

from loguru import logger


def mse(object, reconst):
    return np.mean((object.ravel() - reconst.ravel()) ** 2)


def relative_error(input, output):
    return (
        100
        * np.sum(np.abs(input.ravel() - output.ravel()) ** 2)
        / np.sum(np.abs(input.ravel()) ** 2)
    )


def psnr(vref, vcmp, dyn=None):
    """Compute Peak Signal to Noise Ratio (PSNR) of two images. The PSNR
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
    dyn : None or int, optional (default None)
      Signal dynamic, either the value to use (e.g. 255 for 8 bit samples) or
      None, in which case the actual range of the reference signal is used

    Returns
    -------
    x : float
      PSNR of `vcmp` with respect to `vref`

    """
    if dyn is None:
        dyn = float(vref.max() - vref.min())
    with np.errstate(divide="ignore"):
        msev = mse(vref, vcmp)
        rt = np.where(msev == 0, float("inf"), dyn / np.sqrt(msev))
    return 20.0 * np.log10(rt)


def sam(vref, vcomp):
    """Spetral angle measure

    See also
    -----
    spectral.spectral_angles of Spectral Python toolbolx (SPy)."""
    return np.where(
        np.sqrt(np.sum(vref**2)) * np.sqrt(np.sum(vcomp**2)) == 0,
        0,
        np.arccos(
            np.sum(vref * vcomp)
            / (np.sqrt(np.sum(vref**2)) * np.sqrt(np.sum(vcomp**2)))
        ),
    )


def ssim(vref, vcomp):
    if skimage_available:
        return measure.compare_ssim(vref, vcomp, full=True)
    logger.error("Scikit image must be installed to compute SSIM.")
    return np.nan


def snr(data, data_wo_noise):
    data_flat = []
    data_wonoise_flat = []
    for i, i_wonoise in zip(data, data_wo_noise):
        data_flat.extend(i.ravel())
        data_wonoise_flat.extend(i_wonoise.ravel())
    data_flat = np.asarray(data_flat)
    data_wonoise_flat = np.asarray(data_wonoise_flat)
    return 10 * np.log10(
        np.nan_to_num(
            np.sum(np.asarray(data_flat**2))
            / np.sum((data_flat - data_wonoise_flat) ** 2)
        )
    )
