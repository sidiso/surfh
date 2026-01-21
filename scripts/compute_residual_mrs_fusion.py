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
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/NGC_7023/Fusion/', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=125, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-nt', '--n_templates', default=4, type=int, help='Number of Templates.')
@click.option('-sd', '--scale_data', default=True, type=bool, help='Scale data from Jy  to Jy/str.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-f', '--filtered_data', default=False, type=bool, help='Use filtered MRS data.')
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


    print(f"ndata shape: {ndata.shape}")

    list_raw_data = MRSModel.get_list_of_data(ndata)
    print(f"List of data length: {len(list_raw_data)}")
    for i, d in enumerate(list_raw_data):
        print(f" Channel {list_chan[i]} data shape: {d.shape}")

    corrected_maps = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_11_MO_4_Temp_14_nit_150_mu_5.00e+06_SD_True/res_x.npy')
    corrected_data = MRSModel.forward(corrected_maps)
    list_corrected_data = MRSModel.get_list_of_data(corrected_data)
    print(f"List of corrected data length: {len(list_corrected_data)}")
    for i, d in enumerate(list_corrected_data):
        print(f" Channel {list_chan[i]} corrected data shape: {d.shape}")   

    log.info(f'Start {method} algorithm')
    # reconstruction.reconstruction_method(MRSModel, ndata, templates, paths["result_path"], hyper_parameter, niter, method, scale_data, data_dict)


if __name__ == "__main__":
    parse_options()