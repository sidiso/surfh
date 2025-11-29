import numpy as np
import os
import matplotlib.pyplot as plt
# from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from rich import print

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Others import context

import click
import logging as log

from astropy.io import fits



@click.command()
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MRS_Simulation_Fusion_nmf.yaml', type=str, help='Configuration file.')
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

    log.info('Load MRS data')
    data_dict = model_creation.load_simulated_mrs_data(config)

    log.info('Create intruments and spectro models')
    instruments = model_creation.create_instruments(data_dict, config) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.create_simulation_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape, manual_alpha_shift=-10, manual_beta_shift=-30)
    # MRSModel = model_creation.tmp_create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape, xshift=1, yshift=-1)

    ndata = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_mrs_1a4b.npy')

    # Make masks
    masks = MRSModel.make_mask(ndata)

    log.info(f'Start {config.reconstruction.method} algorithm')
    reconstruction.reconstruction_MRS_fusion(MRSModel, ndata, templates, config, True, data_dict, masks=masks)



    
if __name__ == "__main__":
    parse_options()