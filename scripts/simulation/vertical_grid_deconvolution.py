import numpy as np
import os
import udft
from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle
from importlib import resources
import click

from surfh.Models import wavelength_mrs, realmiri, instru, spectroModel
from surfh.Simulation.fusion_CT import QuadCriterion_MRS




def load_simulation_data(paths, step_angle, Npix):
    """Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle
    origin_beta_axis = np.arange(imshape[1]) * step_angle
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)
    
    
    wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy'))
    spsf = np.load(os.path.join(paths['psf_dir'], 'psfs_pixscale0.05_npix_251_fov12.55_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy'))

    sotf = udft.ir2fr(spsf, imshape)

    return origin_alpha_axis, origin_beta_axis, wavel_axis, sotf




def create_instruments():
    """Create instrument configurations for each channel."""
    instruments = {}

    channel_specs = {
        '1a': (21, 3320, 3710, 0.196, 3.2/3600, 3.7/3600, 8.4),
        '1b': (21, 3190, 3750, 0.196, 3.2/3600, 3.7/3600, 8.4),
        '1c': (21, 3100, 3610, 0.196, 3.2/3600, 3.7/3600, 8.4),
        '2a': (17, 2990, 3110, 0.196, 4.0/3600, 4.8/3600, 8.1),
        '2b': (17, 2750, 3170, 0.196, 4.0/3600, 4.8/3600, 8.1),
        '2c': (17, 2860, 3300, 0.196, 4.0/3600, 4.8/3600, 8.1),
        '3a': (16, 2530, 2880, 0.245, 5.2/3600, 6.2/3600, 7.7),
        '3b': (16, 1790, 2640, 0.245, 5.2/3600, 6.2/3600, 7.7),
        '3c': (16, 1980, 2790, 0.245, 5.2/3600, 6.2/3600, 7.7),
        '4a': (12, 1460, 1930, 0.273, 6.6/3600, 7.7/3600, 8.3),
        '4b': (12, 1680, 1760, 0.273, 6.6/3600, 7.7/3600, 8.3),
        '4c': (12, 1630, 1330, 0.273, 6.6/3600, 7.7/3600, 8.3)
    }

    for chan, (n_slit, r_min, r_max, det_pix_size, fov_x, fov_y, rot_angle) in channel_specs.items():
        spec_blur = instru.SpectralBlur(np.mean([r_min, r_max]))
        instruments[chan] = instru.IFU(
            fov=instru.FOV(fov_x, fov_y, origin=instru.Coord(0, 0), angle=rot_angle),
            det_pix_size=det_pix_size,
            n_slit=n_slit,
            w_blur=spec_blur,
            pce=None,
            wavel_axis=wavelength_mrs.get_mrs_wavelength(chan),
            name=chan.upper()
        )

    return instruments


def create_spectroModel(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings):
    """Create the spectrograph model."""

    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=None,
        alpha_axis=origin_alpha_axis,
        beta_axis=origin_beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)


def create_skyModel(Npix, nwavel):
    # Création d'un cube d'images avec un fond uniforme (par exemple, gris moyen)
    background_color = 128  # Gris moyen
    # Réinitialisation du cube d'images avec un fond uniforme
    cube = np.full((nwavel, Npix, Npix), background_color, dtype=np.uint8)

    # Remplissage de la moitié du cube d'images avec un offset
    offset = 50
    cube[:, :Npix//2, :] = background_color + offset
    # Remplissage de l'autre moitié du cube d'images avec un offset négatif
    cube[:, Npix//2:, :] = background_color - offset
    return cube


def get_dithering(step_Angle):
    with resources.path("surfh.data", "mrs_recommended_dither.dat") as path:
        dithering = np.loadtxt(path, delimiter=",")

    ch1_dither = instru.CoordList.from_array(dithering[:8, :])
    ch2_dither = instru.CoordList.from_array(dithering[8:16, :])
    ch3_dither = instru.CoordList.from_array(dithering[16:24, :])
    ch4_dither = instru.CoordList.from_array(dithering[24:, :])

    main_pointing = instru.Coord(0, 0)    
    pointings = []
    for i in range(3):
        pointing_chan1 = [main_pointing + ch1_dither[i*2]*step_Angle*3 for i in range(4)]
        pointings.append(instru.CoordList(pointing_chan1).pix(step_Angle))
    for i in range(3):
        pointing_chan2 = [main_pointing + ch2_dither[i*2]*step_Angle*3 for i in range(4)]
        pointings.append(instru.CoordList(pointing_chan2).pix(step_Angle))
    for i in range(3):
        pointing_chan3 = [main_pointing + ch3_dither[i*2]*step_Angle*3 for i in range(4)]
        pointings.append(instru.CoordList(pointing_chan3).pix(step_Angle))
    for i in range(3):
        pointing_chan4 = [main_pointing + ch4_dither[i*2]*step_Angle*3 for i in range(4)]
        pointings.append(instru.CoordList(pointing_chan4).pix(step_Angle))
    
    
    return pointings    


def initialize_parameters(fusion_dir_path):
    """Initialize global parameters."""
    paths = {
        'psf_dir': os.path.join(fusion_dir_path, 'PSF/'),
        'template_dir': os.path.join(fusion_dir_path, 'Templates/'),
        'save_filter_corrected_dir': os.path.join(fusion_dir_path, 'Filtered_slices/'),
        'result_path': os.path.join(fusion_dir_path, 'Results/'),
        'mask_path': os.path.join(fusion_dir_path, 'Masks/')
    }
    step = 0.05  # arcsec
    step_angle = Angle(step, u.arcsec).degree

    return paths, step, step_angle

def reconstruction_method(spectroModel, ndata, result_path, hyperParameter, niter, method):
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
    result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_4__nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}/'
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


    # Save results
    print(f"Results save in {path}")
    np.save(path / 'res_x.npy', res_fusion.x)
    np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)



@click.command()
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/Orion_bar/Fusion/', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=251, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-v', '--verbose', default=True, type=bool, help='Verbose.')
def parse_options(fusion_dir, npix, hyper_parameter, niter, method, verbose):

    paths, step, step_angle = initialize_parameters(fusion_dir)

    Npix = 251
    origin_alpha_axis, origin_beta_axis, wavel_axis, sotf = load_simulation_data(paths, step_angle, Npix)

    instruments = create_instruments()
    pointings = get_dithering(step_angle)

    sim_cube = create_skyModel(Npix, len(wavel_axis))

    spectroModel = create_spectroModel(sotf, None, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings)

    sim_data = spectroModel.forward(sim_cube)
    
    reconstruction_method(spectroModel, sim_data, paths['result_path'], hyper_parameter, niter, method)


if __name__ == '__main__':
    parse_options()