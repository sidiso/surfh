import matplotlib.pyplot as plt
import numpy as np
import time
import os
import udft
from pathlib import Path

from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import Angle

import scipy
from scipy.signal import convolve2d as conv2

from surfh import instru, models
from surfh import utils
from surfh import realmiri
from surfh import cython_2D_interpolation


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
step_Angle = Angle(step, u.arcsec)



fits_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_fits'
numpy_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_numpy'#
slices_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_numpy_slices/'

"""
Set Cube coordinate.
"""
margin=100
maps_shape = (maps.shape[0], maps.shape[1]+margin*2, maps.shape[2]+margin*2)
step_Angle = Angle(step, u.arcsec)
origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)

tpl_ss = 3
wavel_axis = wavel_axis[::tpl_ss]
spsf = utils.gaussian_psf(wavel_axis, step_Angle.degree)

if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps_shape[1:])

"""
Process Metadata for all Fits in directory
"""
main_pointing = instru.Coord(0, 0) # Set the main pointing from the instrument FoV
pointings = instru.CoordList([main_pointing])#.pix(step_Angle.degree) # We do not use dithering for first tests
channels = []

filename = 'ChannelCube_ch_1_medium_s3d_0210j_00001'

chan = realmiri.get_IFU('/home/nmonnier/Data/JWST/Orion_bar/All_bands_fits/'+filename+'.fits')
channels.append(chan)
cube = np.load('/home/nmonnier/Data/JWST/Orion_bar/All_bands_numpy/'+filename+'.npy')
cube[np.where(np.isnan(cube))] = 0

origin_alpha_axis += channels[0].fov.origin.alpha
origin_beta_axis += channels[0].fov.origin.beta

spectro = models.Spectro(
    channels, # List of channels and bands 
    origin_alpha_axis, # Alpha Coordinates of the cube
    origin_beta_axis, # Beta Coordinates of the cube
    wavel_axis, # Wavelength axis of the cube
    sotf, # Optical PSF
    pointings, # List of pointing (mainly used for dithering)
    verbose=True,
    serial=False,
)

cube[np.where(cube == 0)] = np.NaN
#slices = spectro.channels[0].realData_cubeToSlice(np.rot90(np.fliplr(cube), 1, (1,2)))
slices = spectro.channels[0].realData_cubeToSlice(cube)

np.save(slices_directory + filename+'.npy', slices)
slices[np.where(np.isnan(slices))] = 0

#slices = spectro.channels[0].realData_cubeToSlice(cube)
ncube = spectro.channels[0].realData_sliceToCube(slices, cube.shape)

plt.figure()
plt.title("Original")
plt.imshow(cube[0])
plt.colorbar()
plt.figure()
plt.imshow(ncube[0])
plt.title("ncube")
plt.colorbar()
plt.show()