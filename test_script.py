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


main_directory = '/home/nmonnier/Data/JWST/Orion_bar/'

fits_directory = main_directory + 'Single_fits/'
numpy_directory = main_directory + 'Single_numpy/'#
slices_directory = main_directory + 'Single_numpy_slices/'


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


"""
Set Cube coordinate.
"""
margin=0
maps_shape = (maps.shape[0], maps.shape[1]+margin*2, maps.shape[2]+margin*2)
step_Angle = Angle(step, u.arcsec)
origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)

tpl_ss = 3
impulse_response = np.ones((1, tpl_ss)) / tpl_ss
tpl = conv2(tpl, impulse_response, "same")[:, ::tpl_ss]
wavel_axis = wavel_axis[::tpl_ss]
spsf = utils.gaussian_psf(wavel_axis, step_Angle.degree)

if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps_shape[1:])
if "sim_cube" not in globals():
    print("Compute sim cube")
    sim_cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)

"""
Process Metadata for all Fits in directory
"""
main_pointing = instru.Coord(0, 0) # Set the main pointing from the instrument FoV
pointings = instru.CoordList([main_pointing])#.pix(step_Angle.degree) # We do not use dithering for first tests
channels = []

filename = 'ChannelCube_ch_1_long_s3d_02111_00001'

chan = realmiri.get_IFU(fits_directory + filename + '.fits')
channels.append(chan)
cube = np.load(numpy_directory + filename + '.npy')
cube[np.where(np.isnan(cube))] = 0

origin_alpha_axis += channels[0].fov.origin.alpha
origin_beta_axis += channels[0].fov.origin.beta

# spectro = models.Spectro(
#     channels, # List of channels and bands 
#     origin_alpha_axis, # Alpha Coordinates of the cube
#     origin_beta_axis, # Beta Coordinates of the cube
#     wavel_axis, # Wavelength axis of the cube
#     sotf, # Optical PSF
#     pointings, # List of pointing (mainly used for dithering)
#     verbose=True,
#     serial=False,
# )

# data = np.load('/home/nmonnier/Data/JWST/Orion_bar/Single_numpy_slices/ChannelCube_ch_1_long_s3d_02111_00001.npy')
# data[np.where(np.isnan(data))] = 0
# test = spectro.sliceToCube(data)

# mask = utils.mask_FoV(test)


spectrolmm = models.SpectroLMM(
    channels, # List of channels and bands 
    origin_alpha_axis, # Alpha Coordinates of the cube
    origin_beta_axis, # Beta Coordinates of the cube
    wavel_axis, # Wavelength axis of the cube
    sotf, # Optical PSF
    pointings, # List of pointing (mainly used for dithering)
    tpl,
    verbose=True,
    serial=True,
)

# print("COmpute Forward")
# data = spectrolmm.forward(maps)

# nnn = spectrolmm.adjoint(data)

# ndata = spectrolmm.forward(nnn)
# sim_cube = spectrolmm.get_cube(nnn)

