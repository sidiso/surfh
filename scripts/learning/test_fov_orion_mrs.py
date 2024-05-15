#!/usr/bin/env python3

""" MIRI MRS Ch3 and Ch4 regarding Orion test case """



import operator as op

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from astropy.io import fits
from scipy.signal import convolve2d as conv2
from surfh.Models import smallmiri as miri


from surfh.Models import instru

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




plt.figure(1)
plt.clf()

plt.plot(
    [
        alpha_axis[0],
        alpha_axis[-1],
        alpha_axis[-1],
        alpha_axis[0],
        alpha_axis[0],
    ],
    [
        beta_axis[0], 
        beta_axis[0], 
        beta_axis[-1], 
        beta_axis[-1], 
        beta_axis[0]
    ],
    "-o",
    label='Orion FoV',
)



plt.plot(
    [
        miri.ch1a.fov.vertices[0].alpha,
        miri.ch1a.fov.vertices[1].alpha,
        miri.ch1a.fov.vertices[2].alpha,
        miri.ch1a.fov.vertices[3].alpha,
        miri.ch1a.fov.vertices[0].alpha,
    ],
    [
        miri.ch1a.fov.vertices[0].beta, 
        miri.ch1a.fov.vertices[1].beta, 
        miri.ch1a.fov.vertices[2].beta, 
        miri.ch1a.fov.vertices[3].beta, 
        miri.ch1a.fov.vertices[0].beta
     ],
    "-x",
    label='ch1a FoV'
)


plt.plot(
    [
        miri.ch2a.fov.vertices[0].alpha,
        miri.ch2a.fov.vertices[1].alpha,
        miri.ch2a.fov.vertices[2].alpha,
        miri.ch2a.fov.vertices[3].alpha,
        miri.ch2a.fov.vertices[0].alpha,
    ],
    [
        miri.ch2a.fov.vertices[0].beta, 
        miri.ch2a.fov.vertices[1].beta, 
        miri.ch2a.fov.vertices[2].beta, 
        miri.ch2a.fov.vertices[3].beta, 
        miri.ch2a.fov.vertices[0].beta
     ],
    "-x",
    label='ch2a FoV',
)

plt.plot(
    [
        miri.ch3a.fov.vertices[0].alpha,
        miri.ch3a.fov.vertices[1].alpha,
        miri.ch3a.fov.vertices[2].alpha,
        miri.ch3a.fov.vertices[3].alpha,
        miri.ch3a.fov.vertices[0].alpha,
    ],
    [
        miri.ch3a.fov.vertices[0].beta, 
        miri.ch3a.fov.vertices[1].beta, 
        miri.ch3a.fov.vertices[2].beta, 
        miri.ch3a.fov.vertices[3].beta, 
        miri.ch3a.fov.vertices[0].beta
     ],
    "-x",
    label='ch3a FoV',
)

plt.plot(
    [
        miri.ch4a.fov.vertices[0].alpha,
        miri.ch4a.fov.vertices[1].alpha,
        miri.ch4a.fov.vertices[2].alpha,
        miri.ch4a.fov.vertices[3].alpha,
        miri.ch4a.fov.vertices[0].alpha,
    ],
    [
        miri.ch4a.fov.vertices[0].beta, 
        miri.ch4a.fov.vertices[1].beta, 
        miri.ch4a.fov.vertices[2].beta, 
        miri.ch4a.fov.vertices[3].beta, 
        miri.ch4a.fov.vertices[0].beta
     ],
    "-x",
    label='ch4a FoV',
)

plt.gca().invert_xaxis()
plt.xlim([-498, -509]) 

plt.legend()
plt.axis("equal")
plt.show()
