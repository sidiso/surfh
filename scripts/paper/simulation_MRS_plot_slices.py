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

    log.info('Create simulation data')
    maps = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_maps.npy')
    cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
    MRSModel.project_FOV()
    # plt.show()
    # raise SystemExit

    # ndata = MRSModel.forward(maps)
    # np.save('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_mrs_1a4b.npy', ndata)
    ndata = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_mrs_1a4b.npy')

    from surfh.Models.wavelength_mrs import get_mrs_wavelength  
    cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
    wavelengeth_cube = wavel_axis
    slice_1a, _ = MRSModel.plot_slice(ndata, 0, 100, dither=[0])
    wavel_1a_slice = get_mrs_wavelength('1a')[100]

    slice_2b, _ = MRSModel.plot_slice(ndata, 4, 100, dither=[0])
    wavel_2b_slice = get_mrs_wavelength('2b')[100]

    slice_3b, _ = MRSModel.plot_slice(ndata, 7, 100, dither=[0])
    wavel_3b_slice = get_mrs_wavelength('3b')[100]

    slice_4b, _ = MRSModel.plot_slice(ndata, 10, 100, dither=[0])
    wavel_4b_slice = get_mrs_wavelength('4b')[100]



    def find_nearest_index(wavelength_axis, wavelength):
        return np.abs(wavelength_axis - wavelength).argmin()

    # Données
    slices = [slice_1a, slice_2b, slice_3b, slice_4b]
    wavelengths = [wavel_1a_slice, wavel_2b_slice, wavel_3b_slice, wavel_4b_slice]
    labels = ["MRS 1A", "MRS 2B", "MRS 3B", "MRS 4B"]
    w_axis = wavelengeth_cube

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })

    fig, axs = plt.subplots(2, 4, figsize=(14, 6),
                            gridspec_kw={'hspace': 0.05, 'wspace': 0.15})

    for i, (slc, wav, lab) in enumerate(zip(slices, wavelengths, labels)):

        idx = find_nearest_index(w_axis, wav)
        cube_slice = cube[idx]

        vmin = min(np.min(slc), np.min(cube_slice))
        vmax = max(np.max(slc), np.max(cube_slice))

        # --------------------------
        # TOP : CUBE SIMULE + CBAR
        # --------------------------
        ax_top = axs[0, i]
        im_top = ax_top.imshow(cube_slice, origin='lower', cmap='inferno',
                            vmin=vmin, vmax=vmax)
        ax_top.set_title(rf"($\lambda = {wav:.3f}\,\mu m$)")

        ax_top.set_xticks([]); ax_top.set_yticks([])

        # Attacher une colorbar COLLÉE à l’image
        div_top = make_axes_locatable(ax_top)
        cax_top = div_top.append_axes("right", size="2%", pad=0.05)  # pad=0 → collée
        cbar_top = fig.colorbar(im_top, cax=cax_top, orientation="vertical")
        cbar_top.ax.tick_params(labelsize=7)

        # --------------------------
        # BOTTOM : RECONSTRUCTION
        # --------------------------
        ax_bottom = axs[1, i]
        im_bottom = ax_bottom.imshow(slc, origin='lower', cmap='inferno',
                                    vmin=vmin, vmax=vmax)
        # ax_bottom.set_title("Reconstruction")
        ax_bottom.set_xticks([]); ax_bottom.set_yticks([])

        div_bottom = make_axes_locatable(ax_bottom)
        cax_bottom = div_bottom.append_axes("right", size="2%", pad=0.05)
        cbar_bottom = fig.colorbar(im_bottom, cax=cax_bottom, orientation="vertical")
        cbar_bottom.ax.tick_params(labelsize=7)
    # plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_mrs_slices.png', dpi=300)
    # plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_mrs_slices.pdf')
    plt.show()
    
if __name__ == "__main__":
    parse_options()