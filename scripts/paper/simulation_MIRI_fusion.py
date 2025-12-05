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
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MIRIM_Simulation_Fusion_nmf.yaml', type=str, help='Configuration file.')
def parse_options(config_file):

    verbose =True
    config = context.Config.from_yaml(config_file)
    config.validate_paths()
    
    maps = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_maps.npy')
    cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
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

    y_mirim= MIRIModel.forward(maps)

    # Make masks

    log.info(f'Start {config.reconstruction.method} algorithm')
    x_maps, x_cube = reconstruction.reconstruction_MIRIM_fusion(MIRIModel, y_mirim, templates, config, True)
    xy_mirim = MIRIModel.forward(x_maps)

    # Plot for a specific index
    index_wavel = 9500

    # Crop 4 pixels on all borders
    crop = 4
    original_slice = cube[index_wavel, crop:-crop, crop:-crop]
    reconstructed_slice = x_cube[index_wavel, crop:-crop, crop:-crop]
    difference_slice = original_slice - reconstructed_slice

    slice_idx = -1
    # For y_mirim (2D) and xy_mirim (2D)
    y_slice = y_mirim[slice_idx, crop:-crop, crop:-crop]
    xy_slice = xy_mirim[slice_idx, crop:-crop, crop:-crop]
    diff_mirim_slice = y_slice - xy_slice

    # Compute vmin/vmax for the first line
    vmin1 = min(original_slice.min(), reconstructed_slice.min())
    vmax1 = max(original_slice.max(), reconstructed_slice.max())

    # Compute vmin/vmax for the second line
    vmin2 = min(y_slice.min(), xy_slice.min())
    vmax2 = max(y_slice.max(), xy_slice.max())

    # Create figure: 2 lines, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ======== FIRST ROW (Cube slices) ========

    im0 = axes[0, 0].imshow(original_slice, vmin=vmin1, vmax=vmax1, cmap='viridis')
    axes[0, 0].set_title('Original Slice (cropped)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(reconstructed_slice, vmin=vmin1, vmax=vmax1, cmap='viridis')
    axes[0, 1].set_title('Reconstructed Slice (cropped)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(difference_slice, cmap='bwr')
    axes[0, 2].set_title('Difference Slice (cropped)')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # ======== SECOND ROW (y_mirim and xy_mirim) ========

    im3 = axes[1, 0].imshow(y_slice, vmin=vmin2, vmax=vmax2, cmap='viridis')
    axes[1, 0].set_title('y_mirim (cropped)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(xy_slice, vmin=vmin2, vmax=vmax2, cmap='viridis')
    axes[1, 1].set_title('xy_mirim (cropped)')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(diff_mirim_slice, cmap='bwr')
    axes[1, 2].set_title('Difference (y - xy) (cropped)')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    parse_options()