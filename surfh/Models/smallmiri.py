#!/usr/bin/env python3


from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger

from surfh.Models import instru


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


logger.debug(f"THIS IS THE SMALL RMS !")

logger.info("Load PCE")

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

res1a = np.mean([3320, 3710]) / 4
res1b = np.mean([3190, 3750]) / 4
res1c = np.mean([3100, 3610]) / 4
res2a = np.mean([2990, 3110]) / 4
res2b = np.mean([2750, 3170]) / 4
res2c = np.mean([2860, 3300]) / 4
res3a = np.mean([2530, 2880]) / 4
res3b = np.mean([1790, 2640]) / 4
res3c = np.mean([1980, 2790]) / 4
res4a = np.mean([1460, 1930]) / 4
res4b = np.mean([1680, 1760]) / 4
res4c = np.mean([1630, 1330]) / 4


logger.info("Sinc² spectral blur")
spec_blur_1a = instru.SpectralBlur(res1a)
spec_blur_1b = instru.SpectralBlur(res1b)
spec_blur_1c = instru.SpectralBlur(res1c)
spec_blur_2a = instru.SpectralBlur(res2a)
spec_blur_2b = instru.SpectralBlur(res2b)
spec_blur_2c = instru.SpectralBlur(res2c)
spec_blur_3a = instru.SpectralBlur(res3a)
spec_blur_3b = instru.SpectralBlur(res3b)
spec_blur_3c = instru.SpectralBlur(res3c)
spec_blur_4a = instru.SpectralBlur(res4a)
spec_blur_4b = instru.SpectralBlur(res4b)
spec_blur_4c = instru.SpectralBlur(res4c)

#%% MRS channel
logger.debug("Instr. CH3 seems wrong.")

ch1a = instru.IFU(
    instru.FOV(3.2, 3.7, origin=instru.Coord(-503.654, -318.742), angle=8.4),
    0.196,
    21,
    spec_blur_1a,
    pce1a,
    wavel_1a,
    "1A",
)
ch1b = instru.IFU(
    instru.FOV(3.2, 3.7, origin=instru.Coord(-503.654, -318.742), angle=8.4),
    0.196,
    21,
    spec_blur_1b,
    pce1b,
    wavel_1b,
    "1B",
)
ch1c = instru.IFU(
    instru.FOV(3.2, 3.7, origin=instru.Coord(-503.654, -318.742), angle=8.4),
    0.196,
    21,
    spec_blur_1c,
    pce1c,
    wavel_1c,
    "1C",
)
ch2a = instru.IFU(
    instru.FOV(4.0, 4.8, origin=instru.Coord(-503.636, -319.091), angle=8.2),
    0.196,
    17,
    spec_blur_2a,
    pce2a,
    wavel_2a,
    "2A",
)
ch2b = instru.IFU(
    instru.FOV(4.0, 4.8, origin=instru.Coord(-503.636, -319.091), angle=8.2),
    0.196,
    17,
    spec_blur_2b,
    pce2b,
    wavel_2b,
    "2B",
)
ch2c = instru.IFU(
    instru.FOV(4.0, 4.8, origin=instru.Coord(-503.636, -319.091), angle=8.2),
    0.196,
    17,
    spec_blur_2c,
    pce2c,
    wavel_2c,
    "2C",
)
ch3a = instru.IFU(
    instru.FOV(5.5, 6.2, origin=instru.Coord(-504.372, -318.798), angle=7.5),
    0.245,
    16,
    spec_blur_3a,
    pce3a,
    wavel_3a,
    "3A",
)
ch3b = instru.IFU(
    instru.FOV(5.5, 6.2, origin=instru.Coord(-504.372, -318.798), angle=7.5),
    0.245,
    16,
    spec_blur_3b,
    pce3b,
    wavel_3b,
    "3B",
)
ch3c = instru.IFU(
    instru.FOV(5.5, 6.2, origin=instru.Coord(-504.372, -318.798), angle=7.5),
    0.245,
    16,
    spec_blur_3c,
    pce3c,
    wavel_3c,
    "3C",
)
ch4a = instru.IFU(
    instru.FOV(6.9, 7.9, origin=instru.Coord(-503.129, -319.488), angle=0),
    0.273,
    12,
    spec_blur_4a,
    pce4a,
    wavel_4a,
    "4A",
)
ch4b = instru.IFU(
    instru.FOV(6.9, 7.9, origin=instru.Coord(-503.129, -319.488), angle=8.3),
    0.273,
    12,
    spec_blur_4b,
    pce4b,
    wavel_4b,
    "4B",
)
ch4c = instru.IFU(
    instru.FOV(6.9, 7.9, origin=instru.Coord(-503.129, -319.488), angle=8.3),
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

ch1_dither = instru.CoordList.from_array(dithering[:8, :])
ch2_dither = instru.CoordList.from_array(dithering[8:16, :])
ch3_dither = instru.CoordList.from_array(dithering[16:24, :])
ch4_dither = instru.CoordList.from_array(dithering[24:, :])


### Local Variables:
### ispell-local-dictionary: "english"
### End:
