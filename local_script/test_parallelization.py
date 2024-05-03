#!/usr/bin/env python3


import itertools as it
import sys
from functools import partial
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import udft
import time
from astropy.io import fits
from scipy.signal import convolve as conv
from scipy.signal import convolve2d as conv2

from surfh.Models import instru
from surfh.Models import smallmiri as miri
from surfh.ToolsDir import utils

from surfh.Models import models
#from surfh.AsyncProcessPool import APP

#%% Init


def orion():
    """Rerturn maps, templates, spatial step and wavelength"""
    maps = fits.open("./cube_orion/abundances_orion.fits")[0].data

    h2_map = maps[0]
    if_map = maps[1]
    df_map = maps[2]
    mc_map = maps[3]

    spectrums = fits.open("./cube_orion/spectra_mir_orion.fits")[1].data
    wavel_axis = spectrums.wavelength

    h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
    if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
    df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
    mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

    return (
        np.asarray((h2_map, if_map, df_map, mc_map)),
        np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
        0.025,
        wavel_axis,
    )


maps, tpl, step, wavel_axis = orion()

spat_ss = 4
ir = np.ones((spat_ss, spat_ss)) / spat_ss ** 2
maps = np.asarray([conv2(arr, ir)[::spat_ss, ::spat_ss] for arr in maps])
step = instru.get_step([chan.det_pix_size for chan in miri.all_chan], 5)
srfs = instru.get_srf(
    [chan.det_pix_size for chan in miri.all_chan],
    step,
)
alpha_axis = np.arange(maps.shape[1]) * step
beta_axis = np.arange(maps.shape[2]) * step
alpha_axis -= np.mean(alpha_axis)
beta_axis -= np.mean(beta_axis)
alpha_axis += +miri.ch1a.fov.origin.alpha
beta_axis += +miri.ch1a.fov.origin.beta

tpl_ss = 3
ir = np.ones((1, tpl_ss)) / tpl_ss
tpl = conv2(tpl, ir, "same")[:, ::tpl_ss]
wavel_axis = wavel_axis[::tpl_ss]

spsf = utils.gaussian_psf(wavel_axis, step)

if "cube" not in globals():
    print("Compute cube")
    cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)
if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps.shape[1:])

#%% Models
main_pointing = instru.Coord(0, 0)
pointings = instru.CoordList(c + main_pointing for c in miri.ch1_dither).pix(step)
# pointings = instru.CoordList([ifu.Coord(5 * step, 5 * step)])

spectro = models.Spectro(
    [miri.ch1a, miri.ch2a],
    alpha_axis,
    beta_axis,
    wavel_axis,
    sotf,
    pointings,
)


# Full sequential time is : 6.75 s ± 2.09 s
#data = spectro.forward(cube)
#data = spectro.forward_cf_parallel(cube)
data = spectro.forward_cf_chan_parallel(cube)

#APP.startWorkers()
""" start4 = time.time()
data4 = spectro.forward_parallel(cube)
end4 = time.time()

start3 = time.time()
data3 = spectro.forward_parallel(cube)
end3 = time.time()

 

print("Forward")
start = time.time()
data = spectro.forward_parallel(cube)
end = time.time()

start2 = time.time()
data2 = spectro.forward_parallel(cube)
end2 = time.time()

print("Total Default Forward 0 time is ", end4-start4)
print("Total Default Forward 1 time is ", end3-start3)
print("Total Parallel Forward time is ", end-start)
print("Total Default Forward time is ", end2-start2)
print(np.allclose(data, data3))
print(np.allclose(data, data2))
print(np.allclose(data2, data3)) """

#APP.terminate()