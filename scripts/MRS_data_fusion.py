import numpy as np
import os
import matplotlib.pyplot as plt

from rich import print

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Others import context

import click
import logging as log



@click.command()
@click.option('-c', '--config_file', default=None, type=str, help='Configuration file.')
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
    instruments = model_creation.create_instruments(data_dict, config.MRS.list_channels) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape)

    data = list()
    for chan in config.MRS.list_channels:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    if True:
        log.info('Data scaling enable')
        ndata = MRSModel.real_data_janskySR_to_jansky(ndata)



    log.info(f'Start {config.reconstruction.method} algorithm')
    reconstruction.reconstruction_MRS_fusion(MRSModel, ndata, templates, config, True, data_dict)


if __name__ == "__main__":
    parse_options()