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

    return origin_alpha_axis, origin_beta_axis, wavel_axis



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


def create_spectroModel(origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings):
    """Create the spectrograph model."""

    return spectroModel.spectroSigRLSCT(
        sotf=None,
        templates=None,
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
    step = 0.05  # arcsec
    step_angle = Angle(step, u.arcsec).degree

    return paths, step, step_angle


def parse_options():

    fusion_dir = '/home/nmonnier/Data/JWST/Simulation/Vertical_grid'

    paths, step, step_angle = initialize_parameters(fusion_dir)

    Npix = 251
    origin_alpha_axis, origin_beta_axis, wavel_axis = load_simulation_data(paths, step_angle, Npix)

    instruments = create_instruments()
    pointings = get_dithering(step_angle)

    sim_cube = create_skyModel(Npix, len(wavel_axis))

    spectroModel = create_spectroModel(origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings)

    res_cube = np.load(fusion_dir + '/Results/lcg_MC_12_MO_4__nit_200_mu_1.00e+02/res_x.npy')
    critertion = np.load(fusion_dir + '/Results/lcg_MC_12_MO_4__nit_200_mu_1.00e+02/criterion.npy')
    plt.figure()
    plt.plot(critertion)
    plt.yscale('log')

    fig, axs = plt.subplots(3, 3)
    fig2, axs2 = plt.subplots(3, 1)
    # Lambda = 6.5um
    fov = spectroModel.channels[1].instr.fov + spectroModel.channels[1].pointings[0]
    ax = axs[0, 0]
    wavel_idx = np.argmin(np.abs(wavel_axis - 6.5))
    im = ax.imshow(sim_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    ax.set_title(r'$\lambda = 6.5\mu m$')
    fig.colorbar(im, ax=ax)
    
    ax = axs[1, 0]
    im = ax.imshow(res_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    fig.colorbar(im, ax=ax)

    ax = axs[2, 0]
    im = ax.imshow(sim_cube[wavel_idx] - res_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=-100, vmax=100)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    fig.colorbar(im, ax=ax)


    ax2 = axs2[0]
    ax2.plot(res_cube[wavel_idx][:,Npix//2], label=r'$\widehat{x}$')
    ax2.plot(sim_cube[wavel_idx][:,Npix//2], label=r'$x$')
    ax2.set_title(r'Central Column $\lambda = 6.5\mu m$')
    ax2.legend()


    # Lambda = 14um
    fov = spectroModel.channels[7].instr.fov + spectroModel.channels[7].pointings[0]
    ax = axs[0, 1]
    wavel_idx = np.argmin(np.abs(wavel_axis - 14))
    im = ax.imshow(sim_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    ax.set_title(r'$\lambda = 14\mu m$')
    fig.colorbar(im, ax=ax)

    ax = axs[1, 1]
    im = ax.imshow(res_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    fig.colorbar(im, ax=ax)

    ax = axs[2, 1]
    im = ax.imshow(sim_cube[wavel_idx] - res_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=-100, vmax=100)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    fig.colorbar(im, ax=ax)

    ax2 = axs2[1]
    ax2.plot(res_cube[wavel_idx][:,Npix//2], label=r'$\widehat{x}$')
    ax2.plot(sim_cube[wavel_idx][:,Npix//2], label=r'$x$')
    ax2.set_title(r'Central Column $\lambda = 14\mu m$')
    ax2.legend()

    # Lambda = 21um
    fov = spectroModel.channels[10].instr.fov + spectroModel.channels[10].pointings[0]
    ax = axs[0, 2]
    wavel_idx = np.argmin(np.abs(wavel_axis - 21))
    im = ax.imshow(sim_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    ax.set_title(r'$\lambda = 21\mu m$')
    fig.colorbar(im, ax=ax)

    ax = axs[1, 2]
    im = ax.imshow(res_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    fig.colorbar(im, ax=ax)

    ax = axs[2, 2]
    im = ax.imshow(sim_cube[wavel_idx] - res_cube[wavel_idx], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=-100, vmax=100)
    ax.plot(
        list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
        list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
        "-x",
        label=f'channel {spectroModel.channels[1].instr.name}'
    )
    fig.colorbar(im, ax=ax)

    ax2 = axs2[2]
    ax2.plot(res_cube[wavel_idx][:,Npix//2], label=r'$\widehat{x}$')
    ax2.plot(sim_cube[wavel_idx][:,Npix//2], label=r'$x$')
    ax2.set_title(r'Central Column $\lambda = 21\mu m$')
    ax2.legend()


    # # Plot real data and projected FoV for each channel 
    # fig, axs = plt.subplots(3, 4)
    # for idx, chan in enumerate(spectroModel.channels):
    #     fov = chan.instr.fov + chan.pointings[0]
    #     ax = axs[idx // 4, idx % 4]
    #     im = ax.imshow(sim_cube[0], extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='gray', vmin=0, vmax=200)
    #     ax.plot(
    #         list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
    #         list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
    #         "-x",
    #         label=f'channel {chan.instr.name}'
    #     )
    #     ax.set_title(f'Channel {chan.instr.name}')
    #     fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == '__main__':
    parse_options()