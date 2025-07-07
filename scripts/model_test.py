import numpy as np
import os
import udft
from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt

from rich import print
from rich.progress import track
from rich.console import Console

from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.Models import spectroModel
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT
from surfh.Preprocessing import model_creation

from aljabr import LinOp, dottest
from scipy import ndimage
from surfh.Simulation.fusion_CT import QuadCriterion_MRS
import argparse
import click
import itertools

import logging as log

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp
from matplotlib.patches import Polygon



@click.command()
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/Orion_bar/Fusion/', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=125, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-nt', '--n_templates', default=4, type=int, help='Number of Templates.')
@click.option('-sd', '--scale_data', default=False, type=bool, help='Scale data from Jy  to Jy/str.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-v', '--verbose', default=False, type=bool, help='Verbose.')
def parse_options(fusion_dir, npix, hyper_parameter, niter, n_templates, scale_data, method, verbose):

    print(f'Options selected are : ') 
    print(f'\t fusion_dir = {fusion_dir}')
    print(f'\t npix = {npix}')
    print(f'\t hyper_parameter = {hyper_parameter}')
    print(f'\t niter = {niter}')
    print(f'\t nTemplates = {n_templates}')
    print(f'\t scale_data = {scale_data}')
    print(f'\t method = {method}')
    print(f'\t verbose = {verbose}')

    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    list_chan = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c']
    imshape = (npix, npix)

    log.info('Initialize basic path parameters')
    step = 0.1  # arcsec
    paths, step_angle = model_creation.initialize_parameters(fusion_dir, step)


    log.info('Load simulation data')
    wavel_axis, templates, sotf = model_creation.load_simulation_data(paths)


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
        if verbose:
            log.info('Data scaling enable')
        ndata = MRSModel.real_data_janskySR_to_jansky(ndata)

    print(len(MRSModel.pointings))
    for pointing in MRSModel.pointings:
        print(f"Pointing list is {pointing}")
    # print(f"spectroModel pointings = {MRSModel.pointings}")
    ndith=[0,1,2,3] 
    chan_idx=3
    slice_idx=-1
    numpy_slice, degrid1, degrid2 = MRSModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)
    chan_idx=4
    numpy_slice, degrid1, degrid3 = MRSModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)


    plt.imshow(degrid2, alpha=0.5)

    plt.imshow(degrid3, alpha=0.8)
    plt.colorbar()
    plt.show()

    # adj1 = MRSModel.adjoint(ndata)
    # fw1  = MRSModel.forward(adj1)
    # adj2 = MRSModel.adjoint(fw1)

    # print(f"Adjoint shape is {adj2.shape}")

    # plt.imshow(adj2[1000])
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    parse_options()