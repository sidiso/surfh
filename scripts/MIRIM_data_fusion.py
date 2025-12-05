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
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MIRIM_Simulation_Fusion_nmf.yaml', type=str, help='Configuration file.')
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
    miri_soft, miri_pce, H_freq, wavel_axis, templates = model_creation.load_miri_simulation_data(config)

    log.info('Create intruments and spectro models')
    MIRIModel = model_creation.create_miri_model(miri_soft, miri_pce, wavel_axis, templates, imshape, step, H_freq)


    log.info('Load MIRIM and MRS data')
    data_mirim = model_creation.load_data_mirim(list_filter, paths["miri_data"])


    log.info('Cerate intruments for spectro and imager   models')
    instruments = model_creation.create_instruments(data_dict, list_chan) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape)
    MIRIModel = model_creation.create_miri_model(miri_soft, miri_pce, wavel_axis, templates, imshape, step, H_freq)


    data = list()
    for chan in list_chan:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    data_miri = list()
    for filt in list_filter:
        data_miri.append(np.array(data_mirim['data'][filt]))
    ndata_mirim = np.concatenate(data_miri)
    print(f"ndata_mirim shape = {ndata_mirim.shape}")
    print(f"miri model ishape = {MIRIModel.ishape}, Miri model oshape = {MIRIModel.oshape}")

    if scale_data:
        log.info('Data scaling enable')
        ndata = MRSModel.real_data_janskySR_to_jansky(ndata)

    log.info(f'Start {method} algorithm')
    # maps = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_11_MO_4_Temp_14_nit_150_mu_5.00e+06_SD_True/res_x.npy')
    # cube = MRSModel.mapsToCube(maps)
    # adj = MIRIModel.adjoint(ndata_mirim)
    # fw = MIRIModel.forward(adj)
    # print(adj.shape)
    # plt.figure()
    # plt.imshow(cube[2500])
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(fw[2])
    # plt.colorbar()
    # plt.show()
    # for i in range(8):
    #     plt.figure()
    #     plt.imshow(ndata_mirim[i])
    #     plt.colorbar()
    # plt.show()
    reconstruction.reconstruction_MIRIM_MRS_method(MRSModel, MIRIModel, ndata, ndata_mirim, templates, paths["result_path"], hyper_parameter, niter, method, scale_data, data_dict)


if __name__ == "__main__":
    parse_options()