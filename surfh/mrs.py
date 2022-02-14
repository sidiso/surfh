#!/usr/bin/env python3


import numpy as np
from astropy.io import fits

from . import ifu


def mrs_pce(filepath):
    """Load PCE of a detector from FITS file"""
    blob = fits.open(filepath)
    return blob[1].data["EFFICIENCY"]


def mrs_wavel_axis(filepath):
    """Load wavelength axis of a detector from FITS file"""
    blob = fits.open(filepath)
    return blob[1].data["WAVELENGTH"]


pce1a = mrs_pce("./data_files/MIRI_FM_MIRIFUSHORT_1SHORT_PCE_07.00.00.fits")
pce1b = mrs_pce("./data_files/MIRI_FM_MIRIFUSHORT_1MEDIUM_PCE_07.00.00.fits")
pce1c = mrs_pce("./data_files/MIRI_FM_MIRIFUSHORT_1LONG_PCE_07.00.00.fits")
pce2a = mrs_pce("./data_files/MIRI_FM_MIRIFUSHORT_2SHORT_PCE_07.00.00.fits")
pce2b = mrs_pce("./data_files/MIRI_FM_MIRIFUSHORT_2MEDIUM_PCE_07.00.00.fits")
pce2c = mrs_pce("./data_files/MIRI_FM_MIRIFUSHORT_2LONG_PCE_07.00.00.fits")
pce3a = mrs_pce("./data_files/MIRI_FM_MIRIFULONG_3SHORT_PCE_07.00.00.fits")
pce3b = mrs_pce("./data_files/MIRI_FM_MIRIFULONG_3MEDIUM_PCE_07.00.00.fits")
pce3c = mrs_pce("./data_files/MIRI_FM_MIRIFULONG_3LONG_PCE_07.00.00.fits")
pce4a = mrs_pce("./data_files/MIRI_FM_MIRIFULONG_4SHORT_PCE_07.00.00.fits")
pce4b = mrs_pce("./data_files/MIRI_FM_MIRIFULONG_4MEDIUM_PCE_07.00.00.fits")
pce4c = mrs_pce("./data_files/MIRI_FM_MIRIFULONG_4LONG_PCE_07.00.00.fits")

wavel_1a = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFUSHORT_1SHORT_PCE_07.00.00.fits")
wavel_1b = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFUSHORT_1MEDIUM_PCE_07.00.00.fits")
wavel_1c = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFUSHORT_1LONG_PCE_07.00.00.fits")
wavel_2a = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFUSHORT_2SHORT_PCE_07.00.00.fits")
wavel_2b = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFUSHORT_2MEDIUM_PCE_07.00.00.fits")
wavel_2c = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFUSHORT_2LONG_PCE_07.00.00.fits")
wavel_3a = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFULONG_3SHORT_PCE_07.00.00.fits")
wavel_3b = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFULONG_3MEDIUM_PCE_07.00.00.fits")
wavel_3c = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFULONG_3LONG_PCE_07.00.00.fits")
wavel_4a = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFULONG_4SHORT_PCE_07.00.00.fits")
wavel_4b = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFULONG_4MEDIUM_PCE_07.00.00.fits")
wavel_4c = mrs_wavel_axis("./data_files/MIRI_FM_MIRIFULONG_4LONG_PCE_07.00.00.fits")

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
ch1a = ifu.Channel(4.2, 3.4, 21, spec_blur_1a, pce1a, wavel_1a, "1A")
ch1b = ifu.Channel(4.2, 3.4, 21, spec_blur_1b, pce1b, wavel_1b, "1B")
ch1c = ifu.Channel(4.2, 3.4, 21, spec_blur_1c, pce1c, wavel_1c, "1C")
ch2a = ifu.Channel(5.1, 4.2, 17, spec_blur_2a, pce2a, wavel_2a, "2A")
ch2b = ifu.Channel(5.1, 4.2, 17, spec_blur_2b, pce2b, wavel_2b, "2B")
ch2c = ifu.Channel(5.1, 4.2, 17, spec_blur_2c, pce2c, wavel_2c, "2C")
ch3a = ifu.Channel(6.4, 5.7, 16, spec_blur_3a, pce3a, wavel_3a, "3A")
ch3b = ifu.Channel(6.4, 5.7, 16, spec_blur_3b, pce3b, wavel_3b, "3B")
ch3c = ifu.Channel(6.4, 5.7, 16, spec_blur_3c, pce3c, wavel_3c, "3C")
ch4a = ifu.Channel(7.2, 7.2, 12, spec_blur_4a, pce4a, wavel_4a, "4A")
ch4b = ifu.Channel(7.2, 7.2, 12, spec_blur_4b, pce4b, wavel_4b, "4B")
ch4c = ifu.Channel(7.2, 7.2, 12, spec_blur_4c, pce4c, wavel_4c, "4C")

#%% Dither
dithering = np.loadtxt("./data_files/mrs_recommended_dither.dat", delimiter=",")

ch1_dither = dithering[:8, :]
ch2_dither = dithering[8:16, :]
ch3_dither = dithering[16:24, :]
ch4_dither = dithering[24:, :]


### Local Variables:
### ispell-local-dictionary: "english"
### End:
