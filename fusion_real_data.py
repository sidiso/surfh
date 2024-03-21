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

from surfh import fusion

from pathlib import Path

import click
import datetime


@click.command()
@click.option('--data_dir', default='/home/nmonnier/Data/JWST/Orion_bar/', type=click.STRING, help='Path to data directory')
@click.option('--res_dir', default='/home/nmonnier/Data/JWST/Orion_bar/fusion_result/', type=click.STRING, help='Path to result directory')
@click.option('--hyper', default=0.1, type=click.FLOAT, help='Hyperparameter')
@click.option('--sim_data', default=True, type=click.BOOL, help='Use simulated or real data')
@click.option('--niter', default=5, type=click.INT, help='Number of iteration for lcg')
@click.option('--multi_chan', default=False, type=click.BOOL, help='Multi-channel or single channel option')
@click.option('--verbose', default=False, type=click.BOOL, help='Set spectro verbose')
def launch_fusion(data_dir, res_dir, hyper, sim_data, niter, multi_chan, verbose):

    if multi_chan is True:
        print("Multi channels/bands fusion")
        fits_directory    = data_dir + 'All_bands_fits/'
        slices_directory  = data_dir + 'All_bands_numpy_slices/'
        psf_directory     = data_dir + 'All_bands_psf/'
    else:
        print("Single channel/band fusion")
        fits_directory      = data_dir + 'Single_fits/'
        slices_directory    = data_dir + 'Single_numpy_slices/'
        psf_directory       = data_dir + 'All_bands_psf/'

    date = str(datetime.date.today())+ '-' + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)
    result_directory = res_dir + date +  '_MC_' + str(multi_chan) + '_SD_' + str(sim_data) + '_NIT_' + str(niter)
    Path(result_directory).mkdir(parents=True, exist_ok=True)    
    print(f"Result dir is {result_directory}")

    # Create result cube
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
    #spsf = utils.gaussian_psf(wavel_axis, step_Angle.degree)
    spsf = np.load(psf_directory + 'psfs_pixscale0.025_fov11.25_date_300123.npy')
    spsf = spsf[:, 100:351, 100:351]
    print("SHAPE PSF ARE ", spsf.shape)

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
    list_channels = []
    list_data = []

    for file in os.listdir(fits_directory):
        # Create IFU for specific fits
        list_channels.append(realmiri.get_IFU(fits_directory + '/' + file))

        # Load and set NaN to 0
        data = np.load(slices_directory + Path(file).stem + '.npy')
        data[np.where(np.isnan(data))] = 0

        list_data.append(data)


    origin_alpha_axis += list_channels[0].fov.origin.alpha
    origin_beta_axis += list_channels[0].fov.origin.beta

    spectro = models.SpectroLMM(
        list_channels, # List of channels and bands 
        origin_alpha_axis, # Alpha Coordinates of the cube
        origin_beta_axis, # Beta Coordinates of the cube
        wavel_axis, # Wavelength axis of the cube
        sotf, # Optical PSF
        pointings, # List of pointing (mainly used for dithering)
        tpl,
        verbose=verbose,
        serial=False,
    )


    if sim_data is True:
        # If simulation data is set, we use simulated data
        print("Simulated data selection")
        y_data = spectro.forward(maps)
    else:
        # Otherwise, we use real data
        print("Real data selection")
        y_data = list_data[0].ravel()


    print("Start LCG ! ")
    quadCrit = fusion.QuadCriterion_MRS(1, np.copy(y_data), spectro, hyper, True, gradient="separated")
    res_lcg = quadCrit.run_lcg(niter, perf_crit = 1)

    """
    Save results
    """
    np.save(result_directory + '/res_x.npy', res_lcg.x)
    np.save(result_directory + '/res_grad_norm.npy', res_lcg.grad_norm)
    np.save(result_directory + '/res_time.npy', res_lcg.time)
    

if __name__ == '__main__':
    launch_fusion()


