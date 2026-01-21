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
from surfh.Vizualisation import cube_vizualisation

import click
import logging as log

from astropy.io import fits



@click.command()
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MIRIM_Fusion_nmf.yaml', type=str, help='Configuration file.')
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
    print(f"MIRI PCE shape = {miri_pce.shape}")

    data_mirim = model_creation.load_data_mirim(config)
    list_y_mirim = list()
    for filt in config.MIRIM.list_filters:
        list_y_mirim.append(np.array(data_mirim['data'][filt]))
    y_mirim = np.concatenate(list_y_mirim)


    log.info(f'Load MIRIM Fusion results')
    path_results = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/MIRIM_lcg_Temp_22_nit_301_mu_1.00e+01_SD_True/'
    x_maps = np.load(os.path.join(path_results, 'res_x.npy'))
    hdul = fits.open(os.path.join(path_results, 'res_cube.fits'))
    cube = hdul[0].data 
    xy_mirim = MIRIModel.forward(x_maps)

    # For each filter, plot data, xy_mirim and residuals where shape of these are (n_filters, npix, npix)
    slice_x = slice(20, -20)
    slice_y = slice(20, -20)
    for i, filt in enumerate(config.MIRIM.list_filters):
        y_data = y_mirim[i, slice_x, slice_y]
        y_model = xy_mirim[i, slice_x, slice_y]
        residuals = y_data - y_model

        pce = miri_pce[i]
        # deconv_filter is the sum of the the PCE multiplyby the cube built from maps and templates at each wavelength

        deconv_filter = np.sum(cube * pce[:, np.newaxis, np.newaxis], axis=0)

        # Mask pixels around the highest value in residuals because they are bas pixels
        max_pos = np.unravel_index(np.argmax(np.abs(residuals)), residuals.shape)
        x_min = max(0, max_pos[0] - 20)
        x_max = min(residuals.shape[0], max_pos[0] + 20)
        y_min = max(0, max_pos[1] - 20)
        y_max = min(residuals.shape[1], max_pos[1] + 20)
        residuals[x_min:x_max, y_min:y_max] = np.nan



        fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        im0 = axes[0, 0].imshow(y_data, origin='lower', cmap='viridis')
        axes[0, 0].set_title(f'Raw Data - Filter {filt}')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(y_model, origin='lower', cmap='viridis')
        axes[0, 1].set_title(f'Model - Filter {filt}')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(residuals, origin='lower', cmap='RdBu_r')
        axes[0, 2].set_title(f'Residuals - Filter {filt}')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(y_data, origin='lower', cmap='viridis')
        axes[1, 0].set_title(f'Raw Data - Filter {filt}')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(deconv_filter, origin='lower', cmap='viridis')
        axes[1, 1].set_title(f'Model - Filter {filt}')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)


        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parse_options()



