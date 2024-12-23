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
from surfh.DottestModels import MCMO_SigRLSCT_Model
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT
from aljabr import LinOp, dottest
from scipy import ndimage
from surfh.Simulation.fusion_CT import QuadCriterion_MRS
import argparse
import click

import logging as log


console = Console()

def load_data(list_chan, save_filter_corrected_dir):
    """Load data for the specified channels."""
    data_dict = {'data': {}, 'target': {}, 'rotation': {}}

    datashape = {
        '1a': (21, 1050, 19), '1b': (21, 1213, 19), '1c': (21, 1400, 19),
        '2a': (17, 970, 24), '2b': (17, 1124, 24), '2c': (17, 1300, 24),
        '3a': (16, 769, 24), '3b': (16, 892, 24), '3c': (16, 1028, 24),
        '4a': (12, 542, 27), '4b': (12, 632, 27), '4c': (12, 717, 27)
    }

    for chan in list_chan:
        data_dict['data'][chan] = []
        data_dict['target'][chan] = []
        data_dict['rotation'][chan] = 0.

    for file in sorted(os.listdir(save_filter_corrected_dir)):
        for chan in list_chan:
            if chan in file:
                data_shape = datashape[chan]
                with fits.open(os.path.join(save_filter_corrected_dir, file)) as hdul:
                    header = hdul[0].header
                    PA_V3 = header['PA_V3']
                    TARG_RA = header['TARG_RA']
                    TARG_DEC = header['TARG_DEC']
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1, 0, 2)

                    data_dict['data'][chan].append(ndata)
                    data_dict['target'][chan].append((TARG_RA, TARG_DEC))
                    data_dict['rotation'][chan] = PA_V3

    return data_dict

def initialize_parameters(fusion_dir_path):
    """Initialize global parameters."""
    paths = {
        'psf_dir': os.path.join(fusion_dir_path, 'PSF/'),
        'template_dir': os.path.join(fusion_dir_path, 'Templates/'),
        'save_filter_corrected_dir': os.path.join(fusion_dir_path, 'Filtered_slices/'),
        'result_path': os.path.join(fusion_dir_path, 'Results/'),
        'mask_path': os.path.join(fusion_dir_path, 'Masks/')
    }
    step = 0.025  # arcsec
    step_angle = Angle(step, u.arcsec).degree

    return paths, step, step_angle

def load_simulation_data(paths, step, step_angle, Npix, nTemplates):
    """Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle - np.mean(np.arange(imshape[0]) * step_angle)
    origin_beta_axis = np.arange(imshape[1]) * step_angle - np.mean(np.arange(imshape[1]) * step_angle)
    
    if nTemplates == 4:
        wavel_file = 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy'
        templates_file = 'nmf_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy'
    elif nTemplates == 6:
        wavel_file = 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_6_templates_SS4.npy'
        templates_file = 'nmf_orion_1ABC_2ABC_3ABC_4ABC_6_templates_SS4.npy'
    else:
        raise NameError("No corresponding Templates name")
    
    wavel_axis = np.load(os.path.join(paths['template_dir'], wavel_file))
    templates = np.load(os.path.join(paths['template_dir'], templates_file))
    spsf = np.load(os.path.join(paths['psf_dir'], 'psfs_pixscale0.025_npix_501_fov12.525_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy'))

    sotf = udft.ir2fr(spsf, imshape)

    templates /= 10e3
    return origin_alpha_axis, origin_beta_axis, wavel_axis, templates, sotf

def create_instruments(data_dict):
    """Create instrument configurations for each channel."""
    instruments = {}

    channel_specs = {
        '1a': (21, 3320, 3710, 0.196, 3.2/3600, 3.7/3600),
        '1b': (21, 3190, 3750, 0.196, 3.2/3600, 3.7/3600),
        '1c': (21, 3100, 3610, 0.196, 3.2/3600, 3.7/3600),
        '2a': (17, 2990, 3110, 0.196, 4.0/3600, 4.8/3600),
        '2b': (17, 2750, 3170, 0.196, 4.0/3600, 4.8/3600),
        '2c': (17, 2860, 3300, 0.196, 4.0/3600, 4.8/3600),
        '3a': (16, 2530, 2880, 0.245, 5.2/3600, 6.2/3600),
        '3b': (16, 1790, 2640, 0.245, 5.2/3600, 6.2/3600),
        '3c': (16, 1980, 2790, 0.245, 5.2/3600, 6.2/3600),
        '4a': (12, 1460, 1930, 0.273, 6.6/3600, 7.7/3600),
        '4b': (12, 1680, 1760, 0.273, 6.6/3600, 7.7/3600),
        '4c': (12, 1630, 1330, 0.273, 6.6/3600, 7.7/3600)
    }

    for chan, (n_slit, r_min, r_max, det_pix_size, fov_x, fov_y) in channel_specs.items():
        spec_blur = instru.SpectralBlur(np.mean([r_min, r_max]))
        instruments[chan] = instru.IFU(
            fov=instru.FOV(fov_x, fov_y, origin=instru.Coord(0, 0), angle=-data_dict['rotation'][chan]),
            det_pix_size=det_pix_size,
            n_slit=n_slit,
            w_blur=spec_blur,
            pce=None,
            wavel_axis=wavelength_mrs.get_mrs_wavelength(chan),
            name=chan.upper()
        )

    return instruments

def create_model(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, data_dict):
    """Create the spectrograph model."""
    main_pointing = instru.Coord(0, 0)

    pointings = []
    for chan in instruments.keys():
        pointing_chan = [main_pointing + instru.Coord(RA, DEC) for RA, DEC in data_dict['target'][chan]]
        pointings.append(instru.CoordList(pointing_chan).pix(step_angle))

    alpha_axis = origin_alpha_axis + data_dict['target']['2a'][2][0]
    beta_axis = origin_beta_axis + data_dict['target']['2a'][2][1]

    return MCMO_SigRLSCT_Model.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)


# Main function to execute the reconstruction method
def reconstruction_method(spectroModel, ndata, templates, result_path, hyperParameter, niter, method, scale_data):
    """
    Perform the reconstruction method and save results.

    Parameters:
        spectroModel: Spectro model object
        ndata: Data array
        templates: Templates array
        pointings: Pointings data
        result_path: Path to save results
        wavel_axis: Wavelength axis array
    """
    # Hyperparameters
    # hyperParameter = 5e3
    # method = "lcg"
    # niter = 50
    value_init = 0

    # Create result directory
    result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_4_Temp_{templates.shape[0]}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}_SD_{scale_data}/'
    path = pathlib.Path(result_path + result_dir)
    path.mkdir(parents=True, exist_ok=True)

    # QuadCriterion initialization
    quadCrit_fusion = QuadCriterion_MRS(
        mu_spectro=1,
        y_spectro=np.copy(ndata),
        model_spectro=spectroModel,
        mu_reg=hyperParameter,
        printing=True,
        gradient="separated"
    )

    # Run the method
    res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit=1, calc_crit=True, value_init=value_init)

    # Convert maps to cube
    y_cube = spectroModel.mapsToCube(res_fusion.x)
    flipped_cube = np.array([np.fliplr(y_cube[i]) for i in range(y_cube.shape[0])])

    # Save results
    print(f"Results save in {path}")
    np.save(path / 'res_x.npy', res_fusion.x)
    np.save(path / 'res_cube.npy', y_cube)
    np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)

    # Optional visualization
    # plot_cube(np.rot90(flipped_cube, -1, axes=(1, 2)), wavel_axis)

# Example of function usage
# reconstruction_method(spectroModel, ndata, templates, pointings, result_path, wavel_axis)




@click.command()
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/Orion_bar/Fusion/', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=501, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-nt', '--n_templates', default=4, type=int, help='Number of Templates.')
@click.option('-sd', '--scale_data', default=False, type=bool, help='Scale data from Jy  to Jy/str.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-v', '--verbose', default=True, type=bool, help='Verbose.')
def parse_options(fusion_dir, npix, hyper_parameter, niter, n_templates, scale_data, method, verbose):

    print(f'Options selected are : ') 
    print(f'\t fusion_dir = {fusion_dir}')
    print(f'\t npix = {npix}')
    print(f'\t hyper_parameter = {hyper_parameter}')
    print(f'\t niter = {niter}')
    print(f'\t nTemplates = {n_templates}')
    print(f'\t scale_data = {scale_data}')
    print(f'\t method = {method}')

    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    list_chan = ['1a', '1b', '1c' , '2a' , '2b' , '2c', '3a', '3b', '3c', '4a', '4b', '4c']

    if verbose:
        log.info('Initialize basic path parameters')
    paths, step, step_angle = initialize_parameters(fusion_dir)

    if verbose:
        log.info('Load simulation data')
    origin_alpha_axis, origin_beta_axis, wavel_axis, templates, sotf = load_simulation_data(paths, step, step_angle, npix, n_templates)

    if verbose:
        log.info('Load MRS data')
    data_dict = load_data(list_chan, paths["save_filter_corrected_dir"])

    if verbose:
        log.info('Cerate intruments and spectro models')
    instruments = create_instruments(data_dict)
    spectroModel = create_model(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, data_dict)

    data = list()
    for chan in list_chan:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    if scale_data:
        ndata = spectroModel.real_data_janskySR_to_jansky(ndata)


    if verbose:
        log.info(f'Start {method} algorithm')
    reconstruction_method(spectroModel, ndata, templates, paths["result_path"], hyper_parameter, niter, method, scale_data)


if __name__ == '__main__':
    parse_options()