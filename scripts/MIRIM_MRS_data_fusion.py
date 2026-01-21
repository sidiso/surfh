import numpy as np
import os
import matplotlib.pyplot as plt

from rich import print

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Others import context
from surfh.Vizualisation import cube_vizualisation
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits

import click
import logging as log



@click.command()
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MRS_MIRIM_Fusion_nmf.yaml', type=str, help='Configuration file.')
def parse_options(config_file):

    verbose =True
    config = context.Config.from_yaml(config_file)
    config.validate_paths()

    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    imshape = (config.cube.npix, config.cube.npix)

    log.info('Initialize basic path parameters')
    step = 0.1
    step_angle = model_creation.initialize_fusion_parameters(config)

    log.info('Load simulation data')
    wavel_axis, templates, sotf = model_creation.load_mrs_simulation_data(config)
    miri_soft, miri_pce, H_freq, wavel_axis, templates = model_creation.load_miri_simulation_data(config)

    log.info('Load MRS data')
    dict_mrs = model_creation.load_mrs_data(config)
    dict_mirim = model_creation.load_data_mirim(config)

    log.info('Create intruments and spectro models')
    instruments = model_creation.create_instruments(dict_mrs, config) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.tmp_create_model(sotf, templates, wavel_axis, instruments, step_angle, dict_mrs, imshape, xshift=1, yshift=-1)
    MIRIModel = model_creation.create_miri_model(miri_soft, miri_pce, wavel_axis, templates, imshape, step, H_freq)

    data = list()
    for chan in config.MRS.list_channels:
        data.append(np.array(dict_mrs['data'][chan]).ravel())
    y_mrs = np.concatenate(data)

    list_y_mirim = list()
    for filt in config.MIRIM.list_filters:
        list_y_mirim.append(np.array(dict_mirim['data'][filt]))
    y_mirim = np.concatenate(list_y_mirim)

    if True:
        log.info('Data scaling enable')
        y_mrs = MRSModel.real_data_janskySR_to_jansky(y_mrs)

    # Make masks
    masks = MRSModel.make_mask(y_mrs)

    log.info(f'Start {config.reconstruction.method} algorithm')
    reconstruction.reconstruction_MIRIM_MRS_method(MRSModel, y_mrs, MIRIModel, y_mirim, templates, config, True, dict_mrs, masks=masks)


if __name__ == "__main__":
    parse_options()