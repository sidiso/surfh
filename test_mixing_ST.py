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

from scipy.signal import convolve2d as conv2

from surfh.ToolsDir import utils
from surfh.ToolsDir import fusion_mixing
from surfh.Models import realmiri
from surfh.Models import mixing
from surfh.Models import instru
from surfh.Models import spectro


from surfh.ToolsDir import fusion_mixing

import aljabr

main_directory = '/home/nmonnier/Data/JWST/Orion_bar/'

fits_directory = main_directory + 'Single_fits/'
numpy_directory = main_directory + 'Single_numpy/'#
slices_directory = main_directory + 'Single_numpy_slices/'
psf_directory     = main_directory + 'All_bands_psf/'

filename = 'ChannelCube_ch_2_short_s3d_0210f_00001'

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

# Each template is normalized by its median
# tpl = np.array([template/np.median(template) for template in tpl])

if "sim_cube" not in globals():
    print("Compute sim cube")
    sim_cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)

main_pointing = instru.Coord(0, 0) # Set the main pointing from the instrument FoV
pointings = instru.CoordList([main_pointing])#.pix(step_Angle.degree) # We do not use dithering for first tests
channels = []
chan = realmiri.get_IFU(fits_directory + filename + '.fits')
channels.append(chan)

origin_alpha_axis += channels[0].fov.origin.alpha
origin_beta_axis += channels[0].fov.origin.beta

spectroModel = spectro.Spectro(
    channels, # List of channels and bands 
    origin_alpha_axis, # Alpha Coordinates of the cube
    origin_beta_axis, # Beta Coordinates of the cube
    wavel_axis, # Wavelength axis of the cube
    None, # Optical PSF
    pointings, # List of pointing (mainly used for dithering)
    verbose=True,
    serial=True,
)


# Generate data projected into the cube
data = np.load('/home/nmonnier/Data/JWST/Orion_bar/Single_numpy_slices/' + filename +'.npy')
data[np.where(np.isnan(data))] = 0
y_cube = spectroModel.sliceToCube(data)

selection_arr = np.where(y_cube < 1e-5)
fast_selection_arr = np.array(np.where(y_cube > 1e-5)).T

STModel = mixing.MixingST(templates=tpl,
                          alpha_axis=origin_alpha_axis,
                          beta_axis=origin_beta_axis,   
                          wavel_axis=wavel_axis,
                          selection_arr=selection_arr,
                          fast_selection_arr=fast_selection_arr)
spectroModel.__del__()
del spectroModel
data = STModel.forward(maps)
# ad_data = STModel.adjoint(data)
# ad_data2 = STModel.c_fast_adjoint(data)

t1 = time.time()
TST1 = STModel.fast_precompute_TST()
e1 = time.time()    
print("Time Cython TST = ", e1-t1)


quadcriterion = fusion_mixing.QuadCriterion_MRS(mu_spectro=1,
                                                y_spectro=y_cube,
                                                model_mixing=STModel,
                                                mu_reg=10e8,
                                                printing=True,
                                                gradient="separated")

res = quadcriterion.run_method('lcg', 5000, value_init=0.5, calc_crit = False)

# def plot_maps_share(true_maps, estimated_maps):
#     n_rows = true_maps.shape[0]
#     n_col = 2
    
#     x = np.copy(estimated_maps)

#     fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharex = True, sharey = True)
#     m = 0
#     n = 0
#     for ax in axes.flat:
#         if (m+n)%2 == 0:
#             ax.imshow(true_maps[m])
#             m += 1
#         else:
#             ax.imshow(x[n])
#             n += 1

def plot_maps(estimated_maps):
    nrow = estimated_maps.shape[0] // 2
    ncols = estimated_maps.shape[0] // 2
    fig, axes = plt.subplots(nrows=nrow, ncols=nrow, sharex = True, sharey = True)

    for i in range(nrow):
        for j in range(ncols):
            m = axes[i,j].imshow(estimated_maps[i*ncols+j])
            fig.colorbar(m, ax=axes[i,j])


# path = '/home/nmonnier/Data/JWST/Orion_bar/Mixing_results/TST/'
# np.save(path + 'init.npy', res.x)
plot_maps(res.x)
plt.show()