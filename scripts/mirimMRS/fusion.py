import numpy as np
import os
import udft
from astropy.io import fits
from astropy.table import Table
import pathlib
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle
from importlib import resources
import click
import statistics
import operator as op
from scipy.interpolate import interp1d

from surfh.Models import wavelength_mrs, realmiri, instru, spectroModel, MiriModel
from surfh.Simulation.fusion_CT import QuadCriterion_MRS
from surfh.Algorithm.criterion_spectroImageur import QuadCriterion_spectroImageur
from surfh.Vizualisation import cube_vizualisation
from surfh.ToolsDir import matrix_op, utils


def load_simulation_metadata_MRS(paths, step_angle, Npix):
    """Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle
    origin_beta_axis = np.arange(imshape[1]) * step_angle
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)
    
    spsf = np.load(os.path.join(paths['psf_dir'], 'psfs_pixscale0.1_npix_150_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy'))

    sotf = udft.ir2fr(spsf, imshape)

    return origin_alpha_axis, origin_beta_axis, sotf


def load_simulation_metadata_MIRIM(paths, step_angle, Npix):
    """ Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle
    origin_beta_axis = np.arange(imshape[1]) * step_angle
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)
    
    spsf = np.load(os.path.join(paths['psf_dir'], 'mirim_psfs_pixscale0.1_npix_150.npy'))
    # sotf = udft.ir2fr(spsf, imshape)
    
    pce = np.load(os.path.join(paths['pce_path'], 'pce.npy'))
    print(pce.shape)
    return origin_alpha_axis, origin_beta_axis, spsf, pce

def create_ifus():
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
    print(f"Create spectroModel with Templates {type(templates)}")
    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=origin_alpha_axis,
        beta_axis=origin_beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)

def load_skyModel(paths):
    def orion():
        """Rerturn maps, templates, spatial step and wavelength"""
        maps = fits.open(os.path.join(paths['template_dir'], "abundances_orion.fits"))[0].data

        h2_map = maps[0]
        if_map = maps[1]
        df_map = maps[2]
        mc_map = maps[3]

        spectrums = fits.open(os.path.join(paths['template_dir'], "spectra_mir_orion.fits"))[1].data
        wavel_axis = spectrums.wavelength

        h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
        if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
        df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
        mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

        return (
            np.asarray((h2_map, if_map, df_map, mc_map)),
            np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
            wavel_axis,
        )
    maps, tpl, wavel_axis = orion()
    return maps, tpl, wavel_axis



def get_dithering(step_Angle, ifus):
    main_pointing = instru.Coord(0, 0)

    pointings = []
    
    ra_ref = -0.00070
    dec_ref = 0.0
    pix_res = 0.2/3600
    ra =  [ra_ref , ra_ref + 4.5*pix_res, ra_ref                 , ra_ref+ 4.5*pix_res]
    dec = [dec_ref, dec_ref             , dec_ref + 4.5*pix_res, dec_ref + 4.5*pix_res]
    for idx, chan in enumerate(ifus.keys()):
        pointing_chan = [main_pointing + instru.Coord(ra[i], dec[i]) for i in range(4)]
        pointings.append(instru.CoordList(pointing_chan).pix(step_Angle))
    
    
    return pointings    


def initialize_parameters(fusion_dir_path, step=0.1):
    """Initialize global parameters."""
    paths = {
        'psf_dir': os.path.join(fusion_dir_path, 'PSF/'),
        'template_dir': os.path.join(fusion_dir_path, 'Templates/'),
        'save_filter_corrected_dir': os.path.join(fusion_dir_path, 'Filtered_slices/'),
        'result_path': os.path.join(fusion_dir_path, 'Results/'),
        'mask_path': os.path.join(fusion_dir_path, 'Masks/'),
        'pce_path': os.path.join(fusion_dir_path, 'PCE/')
    }
    step_angle = Angle(step, u.arcsec).degree

    return paths, step_angle

def reconstruction_method(imageurModel, mirim_data, spectroModel, mrs_data, result_path, hyperParameter, niter, method, bool_templates):
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
    result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_4_lmm_{True}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}/'
    path = pathlib.Path(result_path + result_dir)
    path.mkdir(parents=True, exist_ok=True)

    quadCrit_fusion = QuadCriterion_spectroImageur(
        mu_imager=1.0,
        y_imager=mirim_data,
        model_imager=imageurModel,
        mu_spectro=1.0,
        y_spectro=mrs_data,
        model_spectro=spectroModel,
        mu_reg=hyperParameter,
        printing=True,
        gradient='separated'
    )

    res_fusion = quadCrit_fusion.run_lcg(maximum_iterations=niter, 
                                         perf_crit=None, 
                                         calc_crit=True, 
                                         value_init=value_init)

    print(f"Results save in {path}")
    # Save results
    if bool_templates is False:
        print("No templates, save only cube")
        np.save(path / 'res_cube.npy', res_fusion.x)
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val_lcg)
    else:
        print("Templates loaded, save templates and cube")
        np.save(path / 'res_x.npy', res_fusion.x)
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val_lcg)
        np.save(path / 'res_cube.npy', spectroModel.mapsToCube(np.array(res_fusion.x)))

    utils.plot_maps(res_fusion.x)
    plt.show()

@click.command()
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/Simulation/Orion', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=150, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-v', '--verbose', default=True, type=bool, help='Verbose.')
def parse_options(fusion_dir, hyper_parameter, niter):

    # Parameters
    step = 0.1 #arsec
    Npix_MRS = 150
    Npix_MIRIM = 150
    wavelength_ss = 4

    # Reconstruction parameters
    paths, step_angle = initialize_parameters(fusion_dir, step)


    # Load simulation data
    maps, tpl, wavel_axis    = load_skyModel(paths)
    tpl = tpl[:, ::wavelength_ss]
    wavel_axis = wavel_axis[::wavelength_ss]
    
    # MIRIM FoV selection
    maps = maps[:,150:300,600:750]


    print(f"shape maps {maps.shape}, shape tpl {tpl.shape}, shape wavel_axis {wavel_axis.shape}")

    alpha_axis_mrs, beta_axis_mrs, soft_mrs = load_simulation_metadata_MRS(paths, step_angle, Npix_MRS)
    alpha_axis_mirim, beta_axis_mirim, spsf_mirim, pce_mirim = load_simulation_metadata_MIRIM(paths, step_angle, Npix_MIRIM)



    ifus = create_ifus()
    mrs_pointing = get_dithering(step_angle, ifus)
    spectroModel = create_spectroModel(soft_mrs, tpl, alpha_axis_mrs, beta_axis_mrs, wavel_axis, ifus, step_angle, mrs_pointing)
    print(f"Shape MRS parameters :")
    print(f"soft : {soft_mrs.shape}")

    print("-------------")
    print(f"Shape of  MiriModel parameters :")
    print(f"spsf_mirim : {spsf_mirim.shape}")
    print(f"pce_mirim : {pce_mirim.shape}")
    print(f"wavel_axis : {wavel_axis.shape}")
    print(f"tpl : {tpl.shape}")
    print(f"imshape : {(Npix_MIRIM, Npix_MIRIM)}")
    print(f"step {step}")
    try:
        H_freq = np.load(os.path.join(paths['template_dir'], "H_freq.npy"))
    except:
        H_freq = None
    imageurModel = MiriModel.Mirim_Model_LMM(spsf_mirim, pce_mirim, wavel_axis, tpl, (Npix_MIRIM, Npix_MIRIM), step, H_freq)
    print(f'Imageur ishape {imageurModel.ishape}, oshape {imageurModel.oshape}')
    


    # Simulate data
    mrs_data = spectroModel.forward(maps)
    mirim_data = imageurModel.forward(maps)
    print(f"Mirim data shape is {mirim_data.shape}")

    reconstruction_method(imageurModel, mirim_data, spectroModel, mrs_data, paths["result_path"], hyper_parameter, niter, 'lcg', True)

    # Define figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    

    # Loop through the four maps
    for i, ax in enumerate(axes):
        im = ax.imshow(
            maps[i],
            extent=[alpha_axis_mirim[0], alpha_axis_mirim[-1], beta_axis_mirim[0], beta_axis_mirim[-1]]
        )
        ax.set_title(f'Maps[{i}]')
        fig.colorbar(im, ax=ax)
        
        # Plot the field of view only for the first subplot
        for chan in spectroModel.channels:
            for pointing_idx in range(4):
                fov = chan.instr.fov + chan.pointings[pointing_idx]
                ax.plot(
                    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
                    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
                    '-x'
                )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parse_options()