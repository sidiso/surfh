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

from surfh.Models import instru
from surfh.ToolsDir import utils
from surfh.Models import realmiri

from surfh.ToolsDir import fusion_spectro
from surfh.ToolsDir import fusion_mixing



from pathlib import Path
import click

import datetime
import aljabr

from surfh.Models import spectro
from surfh.Models import spectrolmm
from surfh.Models import mixing

@click.command()
@click.option('--data_dir', default='/home/nmonnier/Data/JWST/Orion_bar/', type=click.STRING, help='Path to data directory')
@click.option('--res_dir', default='/home/nmonnier/Data/JWST/Orion_bar/fusion_result/small/', type=click.STRING, help='Path to result directory')
@click.option('--hyper', default=0.1, type=click.FLOAT, help='Hyperparameter')
@click.option('--sim_data', default=True, type=click.BOOL, help='Use simulated or real data')
@click.option('--niter', default=5, type=click.INT, help='Number of iteration for lcg')
@click.option('--multi_chan', default=False, type=click.BOOL, help='Multi-channel or single channel option')
@click.option('--verbose', default=False, type=click.BOOL, help='Set spectro verbose')
@click.option('--method', default='lcg', type=click.STRING)
@click.option('--margin', default=0, type=click.INT)
@click.option('--value_init', default=100, type=click.FLOAT)
@click.option('--norm', default=True, type=click.BOOL)
def launch_fusion(data_dir, res_dir, hyper, sim_data, niter, multi_chan, verbose, method, margin, value_init, norm):

    print("#################################################")
    print("#################################################")
    print("######## SMALL FUSION TESTING ! #################")
    print("#################################################")
    print("#################################################")

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
        mask_directory      = data_dir + 'Single_mask/'

    if norm :
        init_directory = '/home/nmonnier/Data/JWST/Orion_bar/Mixing_results/TST_small/Norm/'
    else:
        init_directory = '/home/nmonnier/Data/JWST/Orion_bar/Mixing_results/TST_small/NotNorm/'


    mu = str(hyper)
    # mu = mu.split('.')
    # if len(mu[0]) == 1:
    #     mu[0] = '00'+ mu[0]
    # elif len(mu[0]) == 2:
    #     mu[0] = '0'+ mu[0]

    # if len(mu[1]) == 1:
    #     mu[1] = mu[1] + '00'
    # elif len(mu[1]) == 2:
    #     mu[1] = mu[1] + '0'
    # mu = mu[0] + mu[1] 

    val = str(value_init)
    val = val.split('.')
    if len(val[0]) == 1:
        val[0] = '00'+ val[0]
    elif len(val[0]) == 2:
        val[0] = '0'+ val[0]

    if len(val[1]) == 1:
        val[1] = val[1] + '00'
    elif len(val[1]) == 2:
        val[1] = val[1] + '0'
    val = val[0] + val[1]

    # Explicitely set sim_data to True
    sim_data = True

    date = str(datetime.date.today())+ '-' + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)
    result_directory = res_dir + 'Fusion_SD_' + str(sim_data) + '_MC_' + str(multi_chan) + f'_{method}_' + '_Norm_' + str(norm) +'_Mu_' + mu + '_Init_' + val + '_Nit_' + str(niter) + '_' + date
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
    # Multiply maps to match the mean values of real data
    maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in maps])
    step_Angle = Angle(step, u.arcsec)


    """
    Set Cube coordinate.
    """
    # margin=0
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
    spsf = spsf[:, (100-margin):(351+margin), (100-margin):(351+margin)]
    
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

        #mask = np.load(mask_directory + Path(file).stem + '.npy')
        #mask = np.ones_like(mask)

    tpl = tpl[:,list_channels[0].wslice(wavel_axis)]
    spsf = spsf[list_channels[0].wslice(wavel_axis),:,:]
    wavel_axis = wavel_axis[list_channels[0].wslice(wavel_axis)]


    if "sotf" not in globals():
        print("Compute SPSF")
        sotf = udft.ir2fr(spsf, maps_shape[1:])
    if "sim_cube" not in globals():
        print("Compute sim cube")
        sim_cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)

    origin_alpha_axis += list_channels[0].fov.origin.alpha
    origin_beta_axis += list_channels[0].fov.origin.beta

    """ 
    Création modèle instrument
    """

    spectroModel = spectrolmm.SpectroLMM(
        list_channels, # List of channels and bands 
        origin_alpha_axis, # Alpha Coordinates of the cube
        origin_beta_axis, # Beta Coordinates of the cube
        wavel_axis, # Wavelength axis of the cube
        sotf, # Optical PSF
        pointings, # List of pointing (mainly used for dithering)
        tpl,
        verbose=verbose,
        serial=True,
        mask=0,
    )


    # If simulation data is set, we use simulated data
    print("Simulated data selection")
    y_data_simulated = spectroModel.forward(maps)
    projected_cube_simulated = spectroModel.sliceToCube(y_data_simulated)

    # selection_arr = np.where(projected_cube_simulated < 1e-5)
    # fast_selection_arr = np.array(np.where(projected_cube_simulated > 1e-5)).T

    # # Création modèle de mélange pour calcul l'init
    # STModel = mixing.MixingST(templates=tpl,
    #                         alpha_axis=origin_alpha_axis,
    #                         beta_axis=origin_beta_axis,   
    #                         wavel_axis=wavel_axis,
    #                         selection_arr=selection_arr,
    #                         fast_selection_arr=fast_selection_arr)

    # # 
    # quadCrit_mixing = fusion_mixing.QuadCriterion_MRS(mu_spectro=1,
    #                                                 y_spectro=projected_cube_simulated,
    #                                                 model_mixing=STModel,
    #                                                 mu_reg=10,
    #                                                 printing=True,
    #                                                 gradient="separated")


    # res_init = quadCrit_mixing.run_method('lcg', 5000, value_init=0.5, calc_crit = False)

    # value_init = res_init.x

    value_init = np.load('/home/nmonnier/Data/JWST/Orion_bar/Mixing_results/TST_small/NotNorm/init.npy')
    np.save(result_directory + '/init.npy', value_init)

    """
    Mis en place de l'algorithme de fusion 
    """
    print("\n-----------------------------------")
    print("Data fusion starting")
    print("-----------------------------------")
    quadCrit_fusion = fusion_spectro.QuadCriterion_MRS(mu_spectro=1, 
                                        y_spectro=np.copy(y_data_simulated), 
                                        model_spectro=spectroModel, 
                                        mu_reg=hyper, 
                                        printing=True, 
                                        gradient="separated")
    
    res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)
    x_cube = spectroModel.get_cube(res_fusion.x)

    """
    Save results
    """
    np.save(result_directory + '/res_x.npy', res_fusion.x)
    np.save(result_directory + '/res_grad_norm.npy', res_fusion.grad_norm)
    np.save(result_directory + '/res_time.npy', res_fusion.time) 
    np.save(result_directory + '/res_crit_val.npy', quadCrit_fusion.L_crit_val)
    np.save(result_directory + '/res_cube.npy', x_cube)



if __name__ == '__main__':
    launch_fusion()
