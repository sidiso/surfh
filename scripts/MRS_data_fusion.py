import numpy as np
import os
import matplotlib.pyplot as plt

from rich import print

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Vizualisation import cube_vizualisation
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits

import click
import logging as log



@click.command()
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/Point_source/Fusion/', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=125, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-nt', '--n_templates', default=4, type=int, help='Number of Templates.')
@click.option('-sd', '--scale_data', default=False, type=bool, help='Scale data from Jy  to Jy/str.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-f', '--filtered_data', default=True, type=bool, help='Use filtered MRS data.')
@click.option('-v', '--verbose', default=False, type=bool, help='Verbose.')
def parse_options(fusion_dir, npix, hyper_parameter, niter, n_templates, scale_data, method, filtered_data, verbose):

    print(f'Options selected are : ') 
    print(f'\t fusion_dir = {fusion_dir}')
    print(f'\t npix = {npix}')
    print(f'\t hyper_parameter = {hyper_parameter}')
    print(f'\t niter = {niter}')
    print(f'\t nTemplates = {n_templates}')
    print(f'\t scale_data = {scale_data}')
    print(f'\t method = {method}')
    print(f'\t filtered_data = {filtered_data}')
    print(f'\t verbose = {verbose}')

    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    list_chan = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b']
    imshape = (npix, npix)

    log.info('Initialize basic path parameters')
    step = 0.1  # arcsec
    paths, step_angle = model_creation.initialize_parameters(fusion_dir, step, filtered_data)


    log.info('Load simulation data')
    wavel_axis, templates, sotf = model_creation.load_simulation_data(paths, list_chan)


    log.info('Load MRS data')
    data_dict = model_creation.load_data(list_chan, paths["save_filter_corrected_dir"])


    log.info('Cerate intruments and spectro models')
    instruments = model_creation.create_instruments(data_dict, list_chan) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape)

    data = list()
    for chan in list_chan:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    if scale_data:
        log.info('Data scaling enable')
        ndata = MRSModel.real_data_janskySR_to_jansky(ndata)



    # ndith = [0,1,2,3]  # Dithers to use for the test
    # chan_idx = 1  # Channel index to test
    # slice_idx = -1  # Last slice index to test
    # _,_,slice_0 = MRSModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)
    # # chan_idx = 2  # Channel index to test
    # # slice_idx = -1  # Last slice index to test
    # # _,_,slice_1 = MRSModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)


    # adj0 = MRSModel.adjoint(ndata)
    # # fw = MRSModel.forward(adj0)
    # # adj1 = MRSModel.adjoint(fw)


    # # alpha_coord, beta_coord = model_creation.get_axis(MRSModel)
    # # extent = [alpha_coord.min(), alpha_coord.max(), beta_coord.min(), beta_coord.max()]
    # plt.figure()
    # # plt.imshow(slice_0, cmap='viridis', alpha=0.5)
    # # plt.imshow(slice_1, cmap='viridis', extent=extent, alpha=0.5)
    # # plt.colorbar()

    # cube_vizualisation.plot_cube(adj0, np.arange(adj0.shape[0]))

    # MRSModel.project_FOV()   

    # plt.show()    
    
    # path_file = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'
    # res_dir = 'Results/lcg_MC_9_MO_4_Temp_6_nit_500_mu_5.00e+06_SD_True/'
    # wavelengths = np.load(path_file + res_dir + 'wavel.npy')

    # metadata = {'PA_V3': data_dict['PA_V3']['1c'], 
    #             'TARG_RA': data_dict['target']['1c'][0], 'TARG_DEC':data_dict['target']['1c'][1], 
    #             'RA_V1': data_dict['targetV1']['1c'][0], 'DEC_V1': data_dict['targetV1']['1c'][1], 
    #             'RA_REF': data_dict['targetREF']['1c'][0], 'DEC_REF': data_dict['targetREF']['1c'][1],
    #             'ALPHA_AXIS':MRSModel.alpha_axis, 'BETA_AXIS':MRSModel.beta_axis, 'WAVELENGTH':wavelengths}

    # for key in metadata.keys():
    #     print(f"{key}: {metadata[key]}")
    # data = np.load(path_file + res_dir + 'res_cube.npy')

    # save_numpy_to_fits(data, metadata, path_file + res_dir + 'res_cube.fits')
    # raise ValueError("STOP")

    from scipy.ndimage import rotate
    masks = MRSModel.make_mask(ndata)
    masks = np.array(masks)
    for i in range(masks.shape[0]):
        masks[i] = rotate(np.flipud(np.fliplr(masks[i])), angle=data_dict['PA_V3']['1a']-360-8.2, reshape=False, order=0)
        
        

    from astropy.io import fits
    # --- Charger le FITS original ---
    with fits.open('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_11_MO_4_Temp_14_nit_150_mu_5.00e+06_SD_True/res_cube_XX2.fits') as hdul:
        hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])  # copie complète    
        data = hdul_copy[0].data
        plt.figure()
        plt.imshow(data[0])
        plt.figure()
        plt.imshow(data[0]*masks[0])
        plt.show()
        # --- Créer une extension avec le masque ---
        # Ici, on enregistre le cube de masques comme image FITS (ImageHDU)
        mask_hdu = fits.ImageHDU(data=masks.astype(np.int16))  # int16 pour économiser de la place
        mask_hdu.header['EXTNAME'] = 'MASKS'
        hdul_copy.append(mask_hdu)
        hdul_copy.writeto('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_11_MO_4_Temp_14_nit_150_mu_5.00e+06_SD_True/res_CubeMask.fits', overwrite=True)




    log.info(f'Start {method} algorithm')
    # reconstruction.reconstruction_method(MRSModel, ndata, templates, paths["result_path"], hyper_parameter, niter, method, scale_data, data_dict)


if __name__ == "__main__":
    parse_options()