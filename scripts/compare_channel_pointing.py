import numpy as np
import os
import matplotlib.pyplot as plt

from rich import print

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Others import context
from surfh.ToolsDir import alignment

import click
import logging as log



@click.command()
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/tmp_config.yaml', type=str, help='Configuration file.')
def parse_options(config_file):

    verbose =True
    config = context.Config.from_yaml(config_file)
    config.validate_paths()
    
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    imshape = (config.cube.npix, config.cube.npix)

    log.info('Initialize basic path parameters')
    step_angle = model_creation.initialize_fusion_parameters(config)

    log.info('Load simulation data')
    wavel_axis, templates, sotf = model_creation.load_mrs_simulation_data(config)

    templates = templates/100

    log.info('Load MRS data')
    data_dict = model_creation.load_mrs_data(config)

    log.info('Cerate intruments and spectro models')
    instruments = model_creation.create_instruments(data_dict, config) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.tmp_create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape, xshift=1, yshift=-1)

    data = list()
    for chan in config.MRS.list_channels:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    multi_cube, list_multi_cube = MRSModel.all_slice_to_cube(ndata)
    print(f"multi_cube shape = {multi_cube.shape}")
    print(wavel_axis)
    print(f"Shape of both cube  : {list_multi_cube[0].shape} and {list_multi_cube[1].shape}")



    alignment.interactive_align(np.nansum(list_multi_cube[0][:], axis=0), np.nansum(list_multi_cube[1][:], axis=0))
    # plt.figure()
    # plt.imshow(np.nansum(list_multi_cube[0][:-200], axis=0), origin='lower', cmap='inferno')
    # plt.colorbar()
    # plt.title('Channel 3C Sum over wavelength')
    # plt.figure()
    # plt.imshow(np.nansum(list_multi_cube[1][:200], axis=0), origin='lower', cmap='inferno')
    # plt.colorbar()
    # plt.title('Channel 4A Sum over wavelength')
    # plt.show()

if __name__ == "__main__":
    parse_options()