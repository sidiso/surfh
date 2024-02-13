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

from surfh import instru, models
#from surfh import smallmiri as miri
from surfh import dummymiri as miri
#from surfh import miri
from surfh import utils
import scipy

import time
import os
#from surfh import shared_dict

#from surfh.AsyncProcessPoolLight import APPL

from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import Angle
# %% 1) INPUTS FOR MODELS

file = "/home/nmonnier/Projects/JWST/MRS/surfh/surfh/data/simulation_data"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
from matplotlib.image import imread
image = imread('/home/nmonnier/Pictures/Lenna_(test_image).jpg')
gray = rgb2gray(image)


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


margin=100
maps_shape = (maps.shape[0], maps.shape[1]+margin*2, maps.shape[2]+margin*2)
step_Angle = Angle(step, u.arcsec)
origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)
""" alpha_axis += +miri.ch1a.fov.origin.alpha
beta_axis += +miri.ch1a.fov.origin.beta """


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


import os
directory = os.fsencode('/home/nmonnier/Data/JWST/Orion_bar/single_file/')
res_dict = {}
for of in range(2,4):
    res_dict[of] = 0
    for file in os.listdir(directory):

        #%% Real data
        #path = '/home/nmonnier/Data/JWST/Orion_bar/single_file/ChannelCube_ch1-short_s3d_02101_00001.fits'
        hdul = fits.open(directory+file)
        cube = hdul[1].data

        im = hdul[1].data


        gray_gradient = np.arange(im[0].shape[0])
        gray_gradient = gray_gradient*255/(im[0].shape[0]-1)
        gray_gradient = np.tile(gray_gradient.reshape((-1, 1)), (1, im[0].shape[1]))

        #im[0] = gray_gradient#gray[:,:29]

        w = wcs.WCS(hdul[1].header)
        zz,xx,yy = np.indices(im.shape)

        a, b, c = w.wcs_pix2world(xx, yy, zz, 1)

        wavelength_axis = c[:,0,0]
        ra_map = a[0,:,:]
        dec_map = b[0,:,:]

        #of = 2 # Oversapling_factor -> To match the size of the model

        of_ra_map = np.zeros(((ra_map.shape[0]-2)*of, (ra_map.shape[1]-2)*of), dtype=ra_map.dtype)
        of_dec_map = np.zeros(((dec_map.shape[0]-2)*of, (dec_map.shape[1]-2)*of), dtype=dec_map.dtype)

        # Oversample im with 0 padding
        # Then dupplicate value to 0 values
        of_im = np.zeros((im.shape[0], im.shape[1]*of, im.shape[2]*of), dtype=im.dtype)
        of_im[:, ::of, ::of] = im
        n_of_im = np.zeros((of_im.shape[0], (of_im.shape[1]-2*of)*of, (of_im.shape[2]-2*of)*of), dtype=im.dtype)
        n_of_im = of_im[:, of:-of, of:-of]
        of_im = n_of_im

        kern = np.ones((of,of))
        kern = np.pad(kern, (of-1,0))
        for l in range(of_im.shape[0]):
            of_im[l] = scipy.signal.convolve2d(of_im[l], kern, 'same')



        ra_step_y = ra_map[5,6] - ra_map[5,5]
        of_ra_step_y = ra_step_y/of
        ra_step_x = ra_map[6,5] - ra_map[5,5]
        of_ra_step_x = ra_step_x/of

        dec_step_y = dec_map[5,6] - dec_map[5,5]
        of_dec_step_y = dec_step_y/of
        dec_step_x = dec_map[6,5] - dec_map[5,5]
        of_dec_step_x = dec_step_x/of

        short_ra_map = np.delete(ra_map, 0, 0)
        short_ra_map = np.delete(short_ra_map, 0, 1)
        short_ra_map = np.delete(short_ra_map, -1, 0)
        short_ra_map = np.delete(short_ra_map, -1, 1)

        short_dec_map = np.delete(dec_map, 0, 0)
        short_dec_map = np.delete(short_dec_map, 0, 1)
        short_dec_map = np.delete(short_dec_map, -1, 0)
        short_dec_map = np.delete(short_dec_map, -1, 1)

        ### 
        for i in range(ra_map.shape[0]-2):
            for j in range(ra_map.shape[1]-2):
                of_ra_map[i*of, j*of] = ra_map[i+1,j+1]

        for i in range(dec_map.shape[0]-2):
            for j in range(dec_map.shape[1]-2):
                of_dec_map[i*of, j*of] = dec_map[i+1,j+1]

        ### Interpolate unknow values
        for i in range(of_ra_map.shape[0]):
            for j in range(of_ra_map.shape[1]):
                if i%of !=0:
                    of_ra_map[i, j] = of_ra_map[i-(i%of), j] + of_ra_step_x*(i%of)
                else:
                    if j%of != 0:
                        of_ra_map[i, j] = of_ra_map[i, j-(j%of)] + of_ra_step_y*(j%of)

        for i in range(of_dec_map.shape[0]):
            for j in range(of_dec_map.shape[1]):
                if i%of !=0:
                    of_dec_map[i, j] = of_dec_map[i-(i%of), j] + of_dec_step_x*(i%of)
                else:
                    if j%of != 0:
                        of_dec_map[i, j] = of_dec_map[i, j-(j%of)] + of_dec_step_y*(j%of)


        wavelength_idx = np.arange(len(wavelength_axis)-1)
        of_im = of_im[:-1] # Because miri wavelength is not the same size as the real data

        TwoD_of_im = of_im[0].ravel()


        ### Interpolation 2D image
        points = np.vstack(
            [
                of_ra_map.ravel(),
                of_dec_map.ravel()         
            ]
            ).T

        alpha_axis = origin_alpha_axis + hdul[1].header['CRVAL1']
        beta_axis = origin_beta_axis + hdul[1].header['CRVAL2']
        xi = np.vstack(
            [
                np.tile(alpha_axis.reshape((1, -1)), (len(beta_axis), 1)).ravel(), 
                np.tile(beta_axis.reshape((-1, 1)), (1, len(alpha_axis))).ravel()
            ]
            ).T

        from scipy.interpolate import griddata
        interpolated_im = griddata(points, TwoD_of_im, xi, method='linear').reshape(maps_shape[1],maps_shape[2])

        print(f"Fits {file}")
        print(f"Original image max is {np.nanmax(im[0])}")
        print(f"Oversampled image max is {np.nanmax(of_im[0])}")
        print(f"Interpolated image max is {np.nanmax(interpolated_im)}\n")

        print(f"Original image min is {np.nanmin(im[0])}")
        print(f"Oversampled image min is {np.nanmin(of_im[0])}")
        print(f"Interpolated image min is {np.nanmin(interpolated_im)}\n")

        print(f"Relative Difference min is {(np.nanmin(im[0])-np.nanmin(interpolated_im))/np.nanmin(im[0]) *100} %")
        print(f"Relative Difference max is {(np.nanmax(im[0])-np.nanmax(interpolated_im))/np.nanmax(im[0]) *100} %")
        res_dict[of] += (np.nanmin(im[0])-np.nanmin(interpolated_im))/np.nanmin(im[0]) *100
        print("\n\n")
        hdul.close()



    """ plt.figure()
    plt.imshow(of_im[0])
    plt.colorbar()
    plt.figure()
    plt.imshow(np.flipud(np.fliplr(interpolated_im.reshape(maps_shape[1],maps_shape[2]))))
    plt.colorbar()
    plt.show() """

print(res_dict)

### Inteprolation 3D image
""" interpolated_cube = np.zeros((len(wavelength_idx)-1, 251, 251), dtype=interpolated_im.dtype)

for l in range(len(wavelength_idx)-1):
    interpolated_cube[l,:,:] = griddata(points, of_im[l].ravel(), xi, method='linear').reshape(251,251) """

