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

from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger

from . import ifu


def mrs_pce(filename):
    """Load PCE of a detector from FITS file"""
    with resources.path("surfh.data", filename) as path:
        blob = fits.open(path)
        return blob[1].data["EFFICIENCY"]


def mrs_wavel_axis(filename):
    """Load wavelength axis of a detector from FITS file"""
    with resources.path("surfh.data", filename) as path:
        blob = fits.open(path)
        return blob[1].data["WAVELENGTH"]


logger.debug("The parameters in this module ARE NOT VALID.")

logger.info("Load pce")

pce1a = mrs_pce("MIRI_FM_MIRIFUSHORT_1SHORT_PCE_07.00.00.fits")
pce1b = mrs_pce("MIRI_FM_MIRIFUSHORT_1MEDIUM_PCE_07.00.00.fits")
pce1c = mrs_pce("MIRI_FM_MIRIFUSHORT_1LONG_PCE_07.00.00.fits")
pce2a = mrs_pce("MIRI_FM_MIRIFUSHORT_2SHORT_PCE_07.00.00.fits")
pce2b = mrs_pce("MIRI_FM_MIRIFUSHORT_2MEDIUM_PCE_07.00.00.fits")
pce2c = mrs_pce("MIRI_FM_MIRIFUSHORT_2LONG_PCE_07.00.00.fits")
pce3a = mrs_pce("MIRI_FM_MIRIFULONG_3SHORT_PCE_07.00.00.fits")
pce3b = mrs_pce("MIRI_FM_MIRIFULONG_3MEDIUM_PCE_07.00.00.fits")
pce3c = mrs_pce("MIRI_FM_MIRIFULONG_3LONG_PCE_07.00.00.fits")
pce4a = mrs_pce("MIRI_FM_MIRIFULONG_4SHORT_PCE_07.00.00.fits")
pce4b = mrs_pce("MIRI_FM_MIRIFULONG_4MEDIUM_PCE_07.00.00.fits")
pce4c = mrs_pce("MIRI_FM_MIRIFULONG_4LONG_PCE_07.00.00.fits")

logger.info("Load wavelength axis")
wavel_1a = mrs_wavel_axis("MIRI_FM_MIRIFUSHORT_1SHORT_PCE_07.00.00.fits")
wavel_1b = mrs_wavel_axis("MIRI_FM_MIRIFUSHORT_1MEDIUM_PCE_07.00.00.fits")
wavel_1c = mrs_wavel_axis("MIRI_FM_MIRIFUSHORT_1LONG_PCE_07.00.00.fits")
wavel_2a = mrs_wavel_axis("MIRI_FM_MIRIFUSHORT_2SHORT_PCE_07.00.00.fits")
wavel_2b = mrs_wavel_axis("MIRI_FM_MIRIFUSHORT_2MEDIUM_PCE_07.00.00.fits")
wavel_2c = mrs_wavel_axis("MIRI_FM_MIRIFUSHORT_2LONG_PCE_07.00.00.fits")
wavel_3a = mrs_wavel_axis("MIRI_FM_MIRIFULONG_3SHORT_PCE_07.00.00.fits")
wavel_3b = mrs_wavel_axis("MIRI_FM_MIRIFULONG_3MEDIUM_PCE_07.00.00.fits")
wavel_3c = mrs_wavel_axis("MIRI_FM_MIRIFULONG_3LONG_PCE_07.00.00.fits")
wavel_4a = mrs_wavel_axis("MIRI_FM_MIRIFULONG_4SHORT_PCE_07.00.00.fits")
wavel_4b = mrs_wavel_axis("MIRI_FM_MIRIFULONG_4MEDIUM_PCE_07.00.00.fits")
wavel_4c = mrs_wavel_axis("MIRI_FM_MIRIFULONG_4LONG_PCE_07.00.00.fits")

# https://jwst-docs.stsci.edu/jwst-observatory-characteristics/jwst-observatory-coordinate-system-and-field-of-regard
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-mrs-field-and-coordinates

res1a = np.mean([3320, 3710])
res1b = np.mean([3190, 3750])
res1c = np.mean([3100, 3610])
res2a = np.mean([2990, 3110])
res2b = np.mean([2750, 3170])
res2c = np.mean([2860, 3300])
res3a = np.mean([2530, 2880])
res3b = np.mean([1790, 2640])
res3c = np.mean([1980, 2790])
res4a = np.mean([1460, 1930])
res4b = np.mean([1680, 1760])
res4c = np.mean([1630, 1330])


logger.info("Sinc spectral blur")
spec_blur_1a = ifu.SpectralBlur(res1a)
spec_blur_1b = ifu.SpectralBlur(res1b)
spec_blur_1c = ifu.SpectralBlur(res1c)
spec_blur_2a = ifu.SpectralBlur(res2a)
spec_blur_2b = ifu.SpectralBlur(res2b)
spec_blur_2c = ifu.SpectralBlur(res2c)
spec_blur_3a = ifu.SpectralBlur(res3a)
spec_blur_3b = ifu.SpectralBlur(res3b)
spec_blur_3c = ifu.SpectralBlur(res3c)
spec_blur_4a = ifu.SpectralBlur(res4a)
spec_blur_4b = ifu.SpectralBlur(res4b)
spec_blur_4c = ifu.SpectralBlur(res4c)

#%% MRS channel
logger.debug("ChannelParam. CH3 seems wrong.")
ch1a = ifu.ChannelParam(
    ifu.FOV(3.2, 3.7, origin=ifu.Coord(-503.654, -318.742), angle=8.4),
    0.196,
    21,
    spec_blur_1a,
    pce1a,
    wavel_1a,
    "1A",
)
ch1b = ifu.ChannelParam(
    ifu.FOV(3.2, 3.7, origin=ifu.Coord(-503.654, -318.742), angle=8.4),
    0.196,
    21,
    spec_blur_1b,
    pce1b,
    wavel_1b,
    "1B",
)
ch1c = ifu.ChannelParam(
    ifu.FOV(3.2, 3.7, origin=ifu.Coord(-503.654, -318.742), angle=8.4),
    0.196,
    21,
    spec_blur_1c,
    pce1c,
    wavel_1c,
    "1C",
)
ch2a = ifu.ChannelParam(
    ifu.FOV(4.0, 4.8, origin=ifu.Coord(-503.636, -319.091), angle=8.1),
    0.196,
    17,
    spec_blur_2a,
    pce2a,
    wavel_2a,
    "2A",
)
ch2b = ifu.ChannelParam(
    ifu.FOV(4.0, 4.8, origin=ifu.Coord(-503.636, -319.091), angle=8.1),
    0.196,
    17,
    spec_blur_2b,
    pce2b,
    wavel_2b,
    "2B",
)
ch2c = ifu.ChannelParam(
    ifu.FOV(4.0, 4.8, origin=ifu.Coord(-503.636, -319.091), angle=8.1),
    0.196,
    17,
    spec_blur_2c,
    pce2c,
    wavel_2c,
    "2C",
)
ch3a = ifu.ChannelParam(
    ifu.FOV(5.5, 6.2, origin=ifu.Coord(-504.372, -318.798), angle=7.7),
    0.245,
    16,
    spec_blur_3a,
    pce3a,
    wavel_3a,
    "3A",
)
ch3b = ifu.ChannelParam(
    ifu.FOV(5.5, 6.2, origin=ifu.Coord(-504.372, -318.798), angle=7.7),
    0.245,
    16,
    spec_blur_3b,
    pce3b,
    wavel_3b,
    "3B",
)
ch3c = ifu.ChannelParam(
    ifu.FOV(5.5, 6.2, origin=ifu.Coord(-504.372, -318.798), angle=7.7),
    0.245,
    16,
    spec_blur_3c,
    pce3c,
    wavel_3c,
    "3C",
)
ch4a = ifu.ChannelParam(
    ifu.FOV(6.9, 7.9, origin=ifu.Coord(-503.129, -319.488), angle=8.3),
    0.273,
    12,
    spec_blur_4a,
    pce4a,
    wavel_4a,
    "4A",
)
ch4b = ifu.ChannelParam(
    ifu.FOV(6.9, 7.9, origin=ifu.Coord(-503.129, -319.488), angle=8.3),
    0.273,
    12,
    spec_blur_4b,
    pce4b,
    wavel_4b,
    "4B",
)
ch4c = ifu.ChannelParam(
    ifu.FOV(6.9, 7.9, origin=ifu.Coord(-503.129, -319.488), angle=8.3),
    0.273,
    12,
    spec_blur_4c,
    pce4c,
    wavel_4c,
    "4C",
)

all_chan = [
    ch1a,
    ch1b,
    ch1c,
    ch2a,
    ch2b,
    ch2c,
    ch3a,
    ch3b,
    ch3c,
    ch4a,
    ch4b,
    ch4c,
]

#%% Dither
with resources.path("surfh.data", "mrs_recommended_dither.dat") as path:
    dithering = np.loadtxt(path, delimiter=",")

ch1_dither = ifu.CoordList.from_array(dithering[:8, :])
ch2_dither = ifu.CoordList.from_array(dithering[8:16, :])
ch3_dither = ifu.CoordList.from_array(dithering[16:24, :])
ch4_dither = ifu.CoordList.from_array(dithering[24:, :])


### Local Variables:
### ispell-local-dictionary: "english"
### End:
