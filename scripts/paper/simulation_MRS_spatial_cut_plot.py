import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from surfh.Preprocessing import model_creation
from surfh.Others import context
from surfh.Models.wavelength_mrs import get_mrs_wavelength  
from matplotlib.gridspec import GridSpec
import click
import logging as log


# ------------------------------------------------------
#  UTILS
# ------------------------------------------------------
def find_nearest_index(wavelength_axis, wavelength):
    return np.abs(wavelength_axis - wavelength).argmin()


def apply_mask(data, masks, wavelengths):
    """Apply a mask to the data cube."""
    ch_limit = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 24.19]
    w_start = w_stop = 0
    for i in range(masks.shape[0]):
        w_stop = np.where(wavelengths < ch_limit[i])[0][-1] + 1
        sl = slice(w_start, w_stop)
        data[sl] *= masks[i]
        w_start = w_stop
    return data


# ------------------------------------------------------
#  MAIN CODE
# ------------------------------------------------------
@click.command()
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MRS_MIRI_Simulation_Fusion_nmf.yaml')
def parse_options(config_file):

    # ----------------------------
    # Load config + model
    # ----------------------------
    config = context.Config.from_yaml(config_file)
    config.validate_paths()
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    step_angle = model_creation.initialize_fusion_parameters(config)
    wavel_axis, templates, sotf = model_creation.load_mrs_simulation_data(config)
    data_dict = model_creation.load_simulated_mrs_data(config)
    instruments = model_creation.create_instruments(data_dict, config)
    imshape = (config.cube.npix, config.cube.npix)

    MRSModel = model_creation.create_simulation_model(
        sotf, templates, wavel_axis, instruments, step_angle,
        data_dict, imshape,
        manual_alpha_shift=-10, manual_beta_shift=-30
    )
    # MRSModel.project_FOV()

    # ----------------------------
    # Load simulation and reconstruction
    # ----------------------------
    cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
    ndata = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_mrs_1a4b.npy')

    # res_dir = '/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Results/lcg_MC_11_MO_4_Temp_13_nit_500_mu_1.00e+05_SD_True/res_cube.fits'
    res_dir = '/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Results/MIRIM_MRS_lcg_MC_11_MO_4_Temp_13_nit_500_mu_1.00e+05_SD_True/res_cube.fits'

    with fits.open(res_dir) as hdul:
        res_cube = hdul[0].data
        mask = hdul['MASKS'].data
        wavelength = hdul['WCS-TABLE'].data['wavelength'][0].squeeze()

    masked_res_cube = apply_mask(res_cube.copy(), mask, wavelength)
    masked_ref_cube = apply_mask(cube.copy(), mask, wavelength)

    wavelengeth_cube = wavel_axis
    slice_1a, _ = MRSModel.plot_slice(ndata, 0, 100, dither=[0])
    slice_1a[slice_1a < 1] = np.nan
    # remove border of FoV which is lower than 30 percent of mean value
    wavel_1a_slice = get_mrs_wavelength('1a')[100]

    slice_2b, _ = MRSModel.plot_slice(ndata, 4, 100, dither=[0])
    slice_2b[slice_2b < 1] = np.nan
    wavel_2b_slice = get_mrs_wavelength('2b')[100]

    slice_3b, _ = MRSModel.plot_slice(ndata, 7, 100, dither=[0])
    slice_3b[slice_3b < 1] = np.nan
    wavel_3b_slice = get_mrs_wavelength('3b')[100]

    slice_4b, _ = MRSModel.plot_slice(ndata, 10, 100, dither=[0])
    slice_4b[slice_4b < 1] = np.nan
    wavel_4b_slice = get_mrs_wavelength('4b')[100]   
     # ----------------------------
    # What to plot ?
    # ----------------------------

    y_cut = 28

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })

    # Données pour les slices instrumentales
    slices = [slice_1a, slice_2b, slice_3b, slice_4b]
    wavelengths = [wavel_1a_slice, wavel_2b_slice, wavel_3b_slice, wavel_4b_slice]
    labels = ["MRS 1A", "MRS 2B", "MRS 3B", "MRS 4B"]

    for slc, wav, lab in zip(slices, wavelengths, labels):

        # Cherche l'indice du cube réel à la même longueur d’onde
        idx = find_nearest_index(wavelengeth_cube, wav)

        real_slice   = cube[idx]
        fusion_slice = masked_res_cube[idx]

        # ========= FIGURE ==========
        fig = plt.figure(figsize=(13,7))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[2.5, 1], hspace=0.05, wspace=0.15)

        # Axes images
        ax_real  = fig.add_subplot(gs[0, 0])
        ax_inst  = fig.add_subplot(gs[0, 1])
        ax_fus   = fig.add_subplot(gs[0, 2])
        # Axe coupe
        ax_cut   = fig.add_subplot(gs[1, :])

        # --------------------------
        #        IMAGES
        # --------------------------
        ims = []
        ims.append(ax_real.imshow(real_slice, origin="lower", cmap="viridis"))
        ax_real.set_title(fr"Ground Truth - $\lambda$ = {wav:.2f} μm")

        ims.append(ax_inst.imshow(slc, origin="lower", cmap="viridis"))
        ax_inst.set_title(fr"{lab} - $\lambda$ = {wav:.2f} μm")

        ims.append(ax_fus.imshow(fusion_slice, origin="lower", cmap="viridis"))
        ax_fus.set_title(fr"Fusion - $\lambda$ = {wav:.2f} μm")

        # Supprime tous les ticks des images
        for ax in [ax_real, ax_inst, ax_fus]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)
            # Ligne de coupe
            ax.axhline(y_cut, color="red", lw=1.2, ls="--")
            ax.axhline(y_cut, color="red", lw=1.0, ls="--")

        # Normalisation commune pour colorbars
        vmin = np.nanmin([im.get_array() for im in ims])
        vmax = np.nanmax([im.get_array() for im in ims])
        for im in ims: im.set_clim(vmin, vmax)

        # Colorbars fines collées à droite
        for ax, im in zip([ax_real, ax_inst, ax_fus], ims):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(im, cax=cax)

        # --------------------------
        #      COUPE SPATIALE
        # --------------------------
        cut_real   = real_slice[y_cut, :]
        cut_inst   = slc[y_cut, :]
        cut_fusion = fusion_slice[y_cut, :]

        # Supprime le premier et dernier pixel non-NaN dans cut_inst
        valid = np.where(~np.isnan(cut_inst))[0]
        if len(valid) > 2:
            cut_inst[valid[0]] = np.nan
            cut_inst[valid[-1]] = np.nan

        # Seuil pour éviter le bruit
        th = 10
        cut_real   = np.where(cut_real > th, cut_real, np.nan)
        cut_inst   = np.where(cut_inst > th, cut_inst, np.nan)
        cut_fusion = np.where(cut_fusion > th, cut_fusion, np.nan)

        valid_range = np.where(~np.isnan(cut_fusion))[0]

        if len(valid_range) > 0:
            x_min, x_max = valid_range[0], valid_range[-1]
            print(f"x_min: {x_min}, x_max: {x_max}")
            x_range = np.arange(x_min, x_max+1)

            ax_cut.plot(x_range, cut_real[x_min:x_max+1],   color="black", lw=1.4, label="Real")
            ax_cut.plot(x_range, cut_inst[x_min:x_max+1],   color="blue",  lw=1.2, ls="--", label=lab)
            ax_cut.plot(x_range, cut_fusion[x_min:x_max+1], color="red",   lw=1.2, ls="--", label="Fusion")

            # ax_cut.set_xlim(0, slc.shape[1])
            ax_cut.set_xlim(x_min, x_max)

            ax_cut.set_ylabel(r"Flux (MJy sr$^{-1}$)")
            ax_cut.set_xlabel("X pixel")
            ax_cut.grid(True, ls="--", alpha=0.4)
            ax_cut.legend(fontsize=9)

        # plt.suptitle(f"Slices @ {wav:.2f} μm  —  {lab}", fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_MRS_cut_y_{y_cut}_{lab}.png', dpi=300)
        # plt.savefig(f'/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_MRS_cut_y_{y_cut}_{lab}.pdf')

        plt.show()


if __name__ == "__main__":
    parse_options()
