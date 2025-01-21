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
from astropy.io import fits
from scipy.signal import convolve as conv
from scipy.signal import convolve2d as conv2

from surfh.Models import instru
from surfh.Models import smallmiri as miri
#from surfh import dummymiri as miri
#from surfh import miri
from surfh.ToolsDir import utils

import time
import os

from surfh.Models import models
#from surfh import shared_dict

#from surfh.AsyncProcessPoolLight import APPL

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

spatial_subsampling = 4
impulse_response = np.ones((spatial_subsampling, spatial_subsampling)) / spatial_subsampling ** 2
maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in maps])
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
impulse_response = np.ones((1, tpl_ss)) / tpl_ss
tpl = conv2(tpl, impulse_response, "same")[:, ::tpl_ss]
wavel_axis = wavel_axis[::tpl_ss]

spsf = utils.gaussian_psf(wavel_axis, step)

if "cube" not in globals():
    print("Compute cube")
    cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)
    #cube = np.pad(cube, ((0,0), (50,50), (50,50)))
if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps.shape[1:])

#%% Models
main_pointing = instru.Coord(0, 0)
pointings = instru.CoordList(c + main_pointing for c in miri.ch1_dither).pix(step)

from importlib import resources
with resources.path("surfh.data", "mrs_recommended_dither.dat") as path:
    dithering = np.loadtxt(path, delimiter=",")
ch1_dither = instru.CoordList.from_array(dithering[:1, :])
main_pointing = instru.Coord(0, 0)
pointings = instru.CoordList(c + main_pointing for c in ch1_dither).pix(step)

# pointings = instru.CoordList([ifu.Coord(5 * step, 5 * step)])
spectro = models.SpectroLMM(
    [miri.ch1a, miri.ch1b, miri.ch1c, miri.ch2a, miri.ch2b, miri.ch2c],#, miri.ch3a, miri.ch3b, miri.ch3c, miri.ch4a, miri.ch4b, miri.ch4c],
    alpha_axis,
    beta_axis,
    wavel_axis,
    sotf,
    pointings,
    tpl,
    verbose=True,
    serial=False,
)


print("\n\n#############################################\n\
      ############## Forward mult Shared APPL ###########\n\
      #############################################")
start = time.time()
print("Forward")
data = spectro.forward(maps)

end = time.time()
print("Total Time Multiproc shared Forward is ", end-start)

""" np.save("Test_data.npy", data)
 """
#print(np.allclose(data, data_multiproc))


print("\n\n#############################################\n\
      ############## Adjoint Seq ###################\n\
      #############################################")
start = time.time()
cubeT = spectro.adjoint(data)
ncube = spectro.sliceToCube(data)
end = time.time()
print("Total Time Adjoint is ", end-start) 


#spectro.close() # Close spectro object and clear the memory

""" print("Adjoint")
cubeT = spectro.adjoint(data)

print("Adjoint 1")
#cube1 = spectro.adjoint(np.ones_like(data))

#print("Coadd")
#coadd, nhit = spectro.qdcoadd(data)

# %%
plt.figure(1)
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(
    cube[300], extent=(beta_axis[0], beta_axis[-1], alpha_axis[-1], alpha_axis[0])
)
for point in pointings:
    for fov in spectro.channels[0].instr.slit_fov:
        fov = fov + point
        plt.plot(
            [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
            [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
            "-o",
            alpha=0.3,
            color="white",
        )
    fov = spectro.channels[0].instr.fov + point
    plt.plot(
        [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
        [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
        "-o",
        alpha=0.7,
        lw=4,
        color="black",
    )
fov = miri.ch3a.fov + point
plt.plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    alpha=0.3,
    color="white",
)
fov = miri.ch4a.fov + point
plt.plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    alpha=0.3,
    color="white",
)
plt.subplot(1, 2, 2)
plt.imshow(
    cube[300], extent=(beta_axis[0], beta_axis[-1], alpha_axis[-1], alpha_axis[0])
)
plt.show()
for point in pointings:
    fov = spectro.channels[0].instr.fov + point
    plt.plot(
        [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
        [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
        "-o",
    ) """
