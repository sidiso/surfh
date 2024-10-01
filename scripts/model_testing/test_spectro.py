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

from surfh import instru
from surfh import smallmiri as miri
#from surfh.Models import miri
from surfh import utils

import time
import os

from surfh import models



def ref_data():
    return np.load('reference_forward_data.npy'), np.load('reference_adjoint_data.npy')


np.random.seed(100)
n_wavelenght = 10892

map_shape = (4,1000,1000)
random_maps = np.random.random(map_shape)

tpl_shape = (4, n_wavelenght)
random_tpl = np.random.random(tpl_shape)

step = 0.025 #arcsec

wave_step = 0.002209590525156078
wavel_axis = np.arange(n_wavelenght)*wave_step + 4.68044



spatial_subsampling = 4
impulse_response = np.ones((spatial_subsampling, spatial_subsampling)) / spatial_subsampling ** 2
maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in random_maps])
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
tpl = conv2(random_tpl, impulse_response, "same")[:, ::tpl_ss]
wavel_axis = wavel_axis[::tpl_ss]

spsf = utils.gaussian_psf(wavel_axis, step)


if "cube" not in globals():
    print("Compute cube")
    cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)
if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps.shape[1:])

from importlib import resources
with resources.path("surfh.data", "mrs_recommended_dither.dat") as path:
    dithering = np.loadtxt(path, delimiter=",")
ch1_dither = instru.CoordList.from_array(dithering[:1, :])
main_pointing = instru.Coord(0, 0)
pointings = instru.CoordList(c + main_pointing for c in ch1_dither).pix(step)


spectro = models.Spectro(
    [miri.ch1a, miri.ch2b],#, miri.ch1b, miri.ch1c, miri.ch2a],#, miri.ch2b, miri.ch2c],#, miri.ch3a, miri.ch3b, miri.ch3c, miri.ch4a, miri.ch4b, miri.ch4c],
    alpha_axis,
    beta_axis,
    wavel_axis,
    sotf,
    pointings,
)

data = spectro.forward(cube)
np.save('reference_forward_data.npy', data)

cubeT = spectro.adjoint(data)
np.save('reference_adjoint_data.npy', cubeT)