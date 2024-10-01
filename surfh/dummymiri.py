#!/usr/bin/env python3

from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger

from . import instru

logger.debug("THIS IS THE DUMMY REAL SIZE RMS !")

logger.info("Make wavelength axis")
wavel_1a = np.arange(4.9, 5.7, 0.00073) # 4.9-5.74µm  Delta = 1146
wavel_1b = np.arange(5.66, 6.63, 0.00084) # 5.66-6.63 µm
wavel_1c = np.arange(6.53, 7.65, 0.00097) # 6.53-7.65 µm
wavel_2a = np.arange(7.51, 8.77, 0.00117) # 7.51-8.77 µm Delta = 1074
wavel_2b = np.arange(8.67, 10.13, 0.00135) # 8.67-10.13µm 
wavel_2c = np.arange(10.02, 11.70, 0.00155) # 10.02-11.70µm
wavel_3a = np.arange(11.55, 13.47, 0.00224) # 11.55-13.47µm Delta = 857
wavel_3b = np.arange(13.34, 15.57, 0.00260) # 13.34-15.57µm
wavel_3c = np.arange(15.41, 17.98, 0.00299) # 15.41-17.98µm
wavel_4a = np.arange(17.70, 20.95, 0.00574) # 17.70-20.95µm Delta = 566
wavel_4b = np.arange(20.69, 24.48, 0.00669) # 20.69-24.48µm
wavel_4c = np.arange(24.19, 27.9, 0.00655) # 24.19-27.9µm


np.random.seed(150)
pce1a = np.random.rand(wavel_1a.size)/10 + 0.5
pce1b = np.random.rand(wavel_1b.size)/10 + 0.5
pce1c = np.random.rand(wavel_1c.size)/10 + 0.5
pce2a = np.random.rand(wavel_2a.size)/10 + 0.5
pce2b = np.random.rand(wavel_2b.size)/10 + 0.5
pce2c = np.random.rand(wavel_2c.size)/10 + 0.5
pce3a = np.random.rand(wavel_3a.size)/10 + 0.5
pce3b = np.random.rand(wavel_3b.size)/10 + 0.5
pce3c = np.random.rand(wavel_3c.size)/10 + 0.5
pce4a = np.random.rand(wavel_4a.size)/10 + 0.5
pce4b = np.random.rand(wavel_4b.size)/10 + 0.5
pce4c = np.random.rand(wavel_4c.size)/10 + 0.5


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
    instru.FOV(4.0, 4.8, origin=instru.Coord(-503.636, -319.091), angle=8.1),
    0.196,
    17,
    spec_blur_2a,
    pce2a,
    wavel_2a,
    "2A",
)
ch2b = instru.IFU(
    instru.FOV(4.0, 4.8, origin=instru.Coord(-503.636, -319.091), angle=8.1),
    0.196,
    17,
    spec_blur_2b,
    pce2b,
    wavel_2b,
    "2B",
)
ch2c = instru.IFU(
    instru.FOV(4.0, 4.8, origin=instru.Coord(-503.636, -319.091), angle=8.1),
    0.196,
    17,
    spec_blur_2c,
    pce2c,
    wavel_2c,
    "2C",
)
ch3a = instru.IFU(
    instru.FOV(5.5, 6.2, origin=instru.Coord(-504.372, -318.798), angle=7.7),
    0.245,
    16,
    spec_blur_3a,
    pce3a,
    wavel_3a,
    "3A",
)
ch3b = instru.IFU(
    instru.FOV(5.5, 6.2, origin=instru.Coord(-504.372, -318.798), angle=7.7),
    0.245,
    16,
    spec_blur_3b,
    pce3b,
    wavel_3b,
    "3B",
)
ch3c = instru.IFU(
    instru.FOV(5.5, 6.2, origin=instru.Coord(-504.372, -318.798), angle=7.7),
    0.245,
    16,
    spec_blur_3c,
    pce3c,
    wavel_3c,
    "3C",
)
ch4a = instru.IFU(
    instru.FOV(6.9, 7.9, origin=instru.Coord(-503.129, -319.488), angle=8.3),
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