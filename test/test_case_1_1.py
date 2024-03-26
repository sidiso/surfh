#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:43:08 2024

@author: dpineau
"""


import os


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from udft import ir2fr
from scipy.signal import convolve2d as conv2
from scipy.ndimage import gaussian_filter

from surfh.Models import smallmiri as miri
from surfh.Models import instru
from surfh.ToolsDir import utils

# %% 1) INPUTS FOR MODELS

file = "/home/nmonnier/Projects/JWST/MRS/surfh/surfh/data/simulation_data"

def min_not_zero(data):
    Ni, Nj = data.shape
    data_min = np.max(data)
    for i in range(Ni):
        for j in range(Nj):
            data_point = data[i, j]
            if data[i, j] != 0.0 and data_point < data_min:
                data_min = data_point
    return data_min


def rescale_0_1(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

def retrieve_spectra_lmm():
    
    # import des spectres LMM
    L_specs = (
        np.load(file+"/lmm_specs.npy") / 1e3
    )  # CHANGEMENT PAR RAPPORT À inputs_for_models_1_1
    # conversion de µJy/arcsec-2 à mJy/arcsec-2
    # print("spectres en mJy/arcsec-2 (nouveau)")

    tpl = np.copy(L_specs)
    
    # CHANGEMENTS PAR RAPPORT À inputs_for_models_1_3:
    for i in range(len(tpl)):
        for j in range(len(tpl[i])):
            if tpl[i][j] > 150:
                tpl[i][j] = (tpl[i][j+1] + tpl[i][j-1]) / 2
    
    return tpl


def abundance_maps_inputs_1_3(a=0, b=5):

    # import true abundance maps
    fname_true_maps = file +"/decimated_abundance_maps_orion.fits"
    fits_cube = fits.open(fname_true_maps)
    true_maps = np.asarray(fits_cube[0].data, dtype=np.float32)[a:b, :250, :]
    true_maps.shape

    shape_target = true_maps[0].shape
    
    
    # MODIFYING ABUNDANCE MAP 1
    
    true_maps[0][true_maps[0] > 0.8] = 0.8

    # MODIFYING ABUNDANCE MAP 4

    n_map = 3

    map4 = true_maps[n_map]
    # plt.imshow(map4)

    d = 20
    i1, j1 = 104, 202
    # star1 = map4[i1 - d : i1 + d, j1 - d : j1 + d]
    i2, j2 = 121, 318
    # star2 = map4[i2 - d : i2 + d, j2 - d : j2 + d]
    i3, j3 = 113, 345
    # star3 = map4[i3 - d : i3 + d, j3 - d : j3 + d]
    # star3.shape

    mask = np.zeros((2 * d, 2 * d))
    mask.shape

    # plt.imshow(star3)

    map4[i1 - d : i1 + d, j1 - d : j1 + d] = mask
    map4[i2 - d : i2 + d, j2 - d : j2 + d] = mask
    map4[i3 - d : i3 + d, j3 - d : j3 + d] = mask

    # plt.imshow(map4)

    # changing values of map 4
    map4[map4 <= 0.35] = 0
    min_not_zero_map_4 = min_not_zero(map4)
    map4[map4 == 0] = min_not_zero_map_4
    map4_rescaled = rescale_0_1(map4)

    map4_rescaled_blurred = gaussian_filter(map4_rescaled, 1.4)
    
    map4_rerescaled = rescale_0_1(map4_rescaled_blurred)
    
    true_maps[n_map] = map4_rerescaled

    return true_maps, shape_target


class Simulation_Inputs():
    def __init__(self, n_wavelength = 300):
        true_maps, shape_target = abundance_maps_inputs_1_3()
        
        
        # paramètres liés à l'objet à reconstruire x
        
        maps = np.copy(true_maps)
        marge = 80
        maps = np.pad(maps, ((0, 0), (marge, marge), (marge, marge)), "reflect")
        shape_target = maps.shape[1:]
        spat_ss = 1
        ir = np.ones((spat_ss, spat_ss)) / spat_ss ** 2
        maps = np.asarray([conv2(arr, ir)[::spat_ss, ::spat_ss] for arr in maps])
        self.maps = maps
        
        tpl = retrieve_spectra_lmm()
        tpl_ss = 3
        ir = np.ones((1, tpl_ss)) / tpl_ss
        tpl = conv2(tpl, ir, "same")[:, ::tpl_ss]
        self.tpl = tpl
        
        wavel_axis = np.linspace(4.68, 28.74, n_wavelength)
        wavel_axis = wavel_axis[::tpl_ss]
        self.wavel_axis = wavel_axis
        
        
        # paramètres liés aux modèles instruments H
        
        step = instru.get_step([chan.det_pix_size for chan in miri.all_chan], 5)
        self.step = step
        self.srfs = instru.get_srf(
            [chan.det_pix_size for chan in miri.all_chan],
            step,
        )
        
        spsf = utils.gaussian_psf(wavel_axis, step)
        self.spsf = spsf
        self.sotf = ir2fr(spsf, shape_target)
        
        
        
        alpha_axis = np.arange(maps.shape[1]) * step
        beta_axis = np.arange(maps.shape[2]) * step
        alpha_axis -= np.mean(alpha_axis)
        beta_axis -= np.mean(beta_axis)
        alpha_axis += +miri.ch1a.fov.origin.alpha
        beta_axis += +miri.ch1a.fov.origin.beta
        
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        
        
        from importlib import resources
        with resources.path("surfh.data", "mrs_recommended_dither.dat") as path:
            dithering = np.loadtxt(path, delimiter=",")
        ch1_dither = instru.CoordList.from_array(dithering[:1, :])
        main_pointing = instru.Coord(0, 0)
        pointings = instru.CoordList(c + main_pointing for c in ch1_dither).pix(step)
        self.pointings = pointings
        
    
    def maps_to_cube(self, maps, tpl): # maps.shape = (K, X, Y), tpl.shape = (K, L)
        assert maps.shape[0] == tpl.shape[0]
        return np.sum(tpl[:, :, np.newaxis, np.newaxis] * maps[:, np.newaxis, :, :], axis=0)
    
    def true_cube(self):
        return self.maps_to_cube(self.maps, self.tpl)
    
    def plot_tpl(self):
        for i in range(self.tpl.shape[0]):
            plt.plot(self.wavel_axis, self.tpl[i], label = "spectrum {}".format(i + 1))
        plt.grid()
        plt.legend()

        plt.xlabel("Wavelength in µm")
        plt.ylabel("mJy / arcsec²")
    
    def plot_map(self, number):
        plt.imshow(self.maps[number])
        plt.title("Abundance map n°"+str(number+1)+" / "+str(self.maps.shape[0])+".")
