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
import operator as op

from surfh.Models import wavelength_mrs, realmiri, instru, spectroModel

def load_simulation_data(paths, step_angle, Npix):
    """Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle
    origin_beta_axis = np.arange(imshape[1]) * step_angle
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)
    
    
    wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy'))
    templates = np.load(os.path.join(paths['template_dir'], 'scaled_templates.npy'))
    return origin_alpha_axis, origin_beta_axis, wavel_axis, templates



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

def load_skyModel(paths):
    """Load the sky model."""
    return np.load(os.path.join(paths['template_dir'], 'sim_cube.npy')), np.load(os.path.join(paths['template_dir'], 'sim_maps.npy'))


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


def create_spectroModel(origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings, templates):
    """Create the spectrograph model."""

    return spectroModel.spectroSigRLSCT(
        sotf=None,
        templates=templates,
        alpha_axis=origin_alpha_axis,
        beta_axis=origin_beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)

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
    step = 0.1  # arcsec
    step_angle = Angle(step, u.arcsec).degree

    return paths, step, step_angle


import numpy as np
import matplotlib.pyplot as plt
import operator as op

def load_data(fusion_dir):
    res_dir = 'Results/lcg_MC_12_MO_4_lmm_False_nit_1000_mu_1.00e+00'
    res_cube = np.load(f'{fusion_dir}/{res_dir}/res_cube.npy')
    critertion = np.load(f'{fusion_dir}/{res_dir}/criterion.npy')
    return res_cube, critertion 

def plot_criterion(criterion):
    plt.figure()
    plt.plot(criterion)
    plt.yscale('log')

def plot_simulation_results(axs, axs2, spectroModel, sim_cube, res_cube, origin_alpha_axis, origin_beta_axis, wavel_axis, Npix):
    wavelengths = [6.5, 14, 21]
    channel_indices = [1, 7, 10]

    for i, (wavelength, channel_idx) in enumerate(zip(wavelengths, channel_indices)):
        
        name = spectroModel.channels[channel_idx].instr.name
        wavel_idx = np.argmin(np.abs(wavel_axis - wavelength))

        plot_image(axs[0, i], sim_cube[wavel_idx], origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx], f'$\lambda = {wavelength}\mu m$')
        plot_image(axs[1, i], np.rot90(np.fliplr(res_cube[wavel_idx]), -1), origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx])
        plot_image(axs[2, i], sim_cube[wavel_idx] - np.rot90(np.fliplr(res_cube[wavel_idx]), -1), origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx])

        # plot_central_column(axs2[i], np.rot90(np.fliplr(res_cube[wavel_idx]), -1), sim_cube[wavel_idx], Npix, wavelength, spectroModel.channels[channel_idx])
        plot_central_column_with_FoV(axs2[i], np.rot90(np.fliplr(res_cube[wavel_idx]), -1), sim_cube[wavel_idx], Npix, wavelength, spectroModel.channels[channel_idx], origin_alpha_axis, origin_beta_axis)

def plot_image(ax, data, origin_alpha_axis, origin_beta_axis, name, chan, title=''):
    im = ax.imshow(data, extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray')

    for i in range(1):
        fov = chan.instr.fov + chan.pointings[i]
        ax.plot(
            list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
            list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
            "-x",
            label=f'channel {name}'
        )
    ax.set_title(title)
    ax.legend()
    ax.figure.colorbar(im, ax=ax)




def plot_central_column(ax, res_cube_slice, sim_cube_slice, Npix, wavelength, chan):
    ax.plot(res_cube_slice[:, Npix//2], label=r'$\widehat{x}$')
    ax.plot(sim_cube_slice[:, Npix//2], label=r'$x$')
    ax.set_title(f'Central Column $\lambda = {wavelength}\mu m$')
    ax.legend()

def plot_central_column_bis(ax, res_cube_slice, sim_cube_slice, Npix, wavelength, method, mu):
    ax.plot(res_cube_slice[:, Npix//2], label=r'$\widehat{x}$')
    ax.plot(sim_cube_slice[:, Npix//2], label=r'$x$')
    ax.set_ylabel(f'{method} $\mu = {mu}$')
    ax.legend()

def plot_central_column_with_FoV(ax, res_cube_slice, sim_cube_slice, Npix, wavelength, chan, origin_alpha_axis, origin_beta_axis):
    intersections = []
    alpha_coupe = origin_alpha_axis[Npix // 2]  

    fov = chan.instr.fov + chan.pointings[0]

    for j in range(len(fov.vertices)):
        alpha1, beta1 = fov.vertices[j].alpha, fov.vertices[j].beta
        alpha2, beta2 = fov.vertices[(j + 1) % len(fov.vertices)].alpha, fov.vertices[(j + 1) % len(fov.vertices)].beta

        if (alpha1 - alpha_coupe) * (alpha2 - alpha_coupe) < 0:  
            t = (alpha_coupe - alpha1) / (alpha2 - alpha1)
            beta_inter = beta1 + t * (beta2 - beta1)
            intersections.append(beta_inter)

    beta_axis = origin_beta_axis  # y-coordinates in physical units

    ax.plot(beta_axis, res_cube_slice[:, Npix//2], label=r'$\widehat{x}$')  
    ax.plot(beta_axis, sim_cube_slice[:, Npix//2], label=r'$x$')  
    for beta in intersections:
        ax.axvline(x=beta, color='red', linestyle='--', alpha=0.7, label="Intersection FoV" if beta == intersections[0] else "")

    ax.set_xlabel(r'$\beta$ coordinate') 
    ax.legend()

def plot_sim_maps(sim_maps, origin_alpha_axis, origin_beta_axis):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  

    extent = [origin_alpha_axis.min(), origin_alpha_axis.max(), 
              origin_beta_axis.min(), origin_beta_axis.max()]  

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(sim_maps[i], extent=extent, origin='lower', cmap='viridis')
        ax.set_title(f"Carte d'abondance {i+1}", fontsize=16)  
        ax.set_xlabel(r"$\alpha$ coordinate", fontsize=14)  
        ax.set_ylabel(r"$\beta$ coordinate", fontsize=14)  

        # Réduction du nombre de ticks sur l'axe x
        ax.set_xticks(np.linspace(origin_alpha_axis.min(), origin_alpha_axis.max(), num=5))  # 5 ticks
        ax.set_yticks(np.linspace(origin_beta_axis.min(), origin_beta_axis.max(), num=5))  # 5 ticks sur y aussi
       
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  

    plt.subplots_adjust(wspace=0.4, hspace=0.3)  

    # save_path = '/home/nmonnier/figure1.pdf'
    # plt.savefig(save_path, dpi=300, bbox_inches='tight', format=save_path.split('.')[-1])  
    # print(f"Image sauvegardée : {save_path}")

    plt.show()



def parse_options():
    fusion_dir = '/home/nmonnier/Data/JWST/Simulation/Cross_grid'

    paths, step, step_angle = initialize_parameters(fusion_dir)
    Npix = 125
    origin_alpha_axis, origin_beta_axis, wavel_axis, templates = load_simulation_data(paths, step_angle, Npix)

    instruments = create_instruments()
    pointings = get_dithering(step_angle)

    _, sim_maps = load_skyModel(paths)

    spectroModel = create_spectroModel(origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings, templates)
    sim_cube = spectroModel.mapsToCube(sim_maps)

    # Appel de la fonction avec tes données
    plot_sim_maps(sim_maps, origin_alpha_axis, origin_beta_axis)


    """
    Here multiple simulations analysis
    """

    # res_dir = fusion_dir + '/Results/'
    # fig, axs = plt.subplots(len(os.listdir(res_dir)), 1)
    # for res_idx, res in enumerate(sorted(os.listdir(res_dir))):
    #     split_res = os.path.splitext(res)[0].split('_')
    #     print(split_res)
    #     method = split_res[0]
    #     mu = split_res[-1]
    #     res_cube = np.load(f'{res_dir}/{res}/res_cube.npy')
    #     wavelength = 21.5
    #     wavel_idx = np.argmin(np.abs(wavel_axis - wavelength))
    #     plot_central_column_bis(axs[res_idx], res_cube[wavel_idx], sim_cube[wavel_idx], Npix, wavelength, method, mu)
    # plt.show()

    """
    Here single simulation
    """
    res_cube, critertion = load_data(fusion_dir)
    plot_criterion(critertion)

    fig, axs = plt.subplots(3, 3)
    fig2, axs2 = plt.subplots(3, 1)
    plot_simulation_results(axs, axs2, spectroModel, sim_cube, res_cube, origin_alpha_axis, origin_beta_axis, wavel_axis, Npix)

    plt.figure()
    plt.plot(sim_cube[:, Npix//2, Npix//2], label='sim')
    plt.plot(res_cube[:, Npix//2, Npix//2], label='res')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parse_options()