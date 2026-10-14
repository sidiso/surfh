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
from surfh.Models import wavelength_mrs


import click
import logging as log

from astropy.io import fits



@click.command()
@click.option('-c', '--config_file', default='/home/nmonnier/Projects/JWST/MRS/surfh/config/config_MIRIM_Fusion_nmf_1B_4C.yaml', type=str, help='Configuration file.')
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
    miri_pce = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/pce_interpolated_mirim_fusion.npy')
    miri_pce = miri_pce[1:]

    log.info('Rescale wavelength, psf, pce to match selected band ans spectral subsampling')

    print("Wavelength axis shape = ", wavel_axis.shape)
    print("PCE shape = ", miri_pce.shape)
    print("PSF shape = ", miri_soft.shape)
    print("Templates shape = ", templates.shape)

    log.info('Create intruments and spectro models')
    H_freq = None
    MIRIModel = model_creation.create_miri_model(miri_soft, miri_pce, wavel_axis, templates, imshape, step, H_freq)

    data_mirim = model_creation.load_data_mirim(config)
    list_y_mirim = list()
    for filt in config.MIRIM.list_filters:
        list_y_mirim.append(np.array(data_mirim['data'][filt]))
    y_mirim = np.concatenate(list_y_mirim)

    print(f"Shape y_mirim = {y_mirim.shape}")


    # raise SystemExit

    # Plot figures to save 
    # coordonnées en arcsec, centrées
    x = (np.arange(imshape[0]) - imshape[0] / 2) * step
    y = (np.arange(imshape[1]) - imshape[1] / 2) * step

    extent = [x[0], x[-1], y[0], y[-1]]

    plt.figure(figsize=(6, 5))
    plt.imshow(y_mirim[0], origin="lower", extent=extent, cmap="viridis", aspect="equal")

    plt.xlabel("ΔRA (arcsec)")
    plt.ylabel("ΔDec (arcsec)")
    plt.title(f"MIRIM Raw Data — Filter {config.MIRIM.list_filters[0]}")

    cbar = plt.colorbar()
    cbar.set_label("Flux MJy/sr")  # adapte selon tes unités

    plt.tight_layout()
    # plt.savefig(f'/home/nmonnier/Presentations/20260123_INCLASS/MIRIM_Filter_{config.MIRIM.list_filters[0]}_Data.png', dpi=200)
    plt.show()




    log.info(f'Start {config.reconstruction.method} algorithm')
    x_maps, x_cube = reconstruction.reconstruction_MIRIM_fusion(MIRIModel, y_mirim, templates, config, True, config_file)
    # x_maps = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/MIRIM_lcg_Temp_12_nit_300_mu_1.00e+01_SD_True/res_x.npy')
    # hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/MIRIM_lcg_Temp_12_nit_300_mu_1.00e+01_SD_True/res_cube.fits')
    # x_cube = hdul[0].data   
    # hdul.close()
    xy_mirim = MIRIModel.forward(x_maps)


    def crop_border(img, ny=10, nx=10):
        return img[ny:-ny, nx:-nx]

    cropx = 20
    cropy = 60
    for i, filt in enumerate(config.MIRIM.list_filters):

        img_fusion = crop_border(xy_mirim[i], cropy, cropx)
        img_orig   = crop_border(y_mirim[i], cropy, cropx)
        img_diff   = np.abs(img_orig - img_fusion)

        ny, nx = img_fusion.shape

        # axes centrés en arcsec
        x = (np.arange(nx) - nx / 2) * step
        y = (np.arange(ny) - ny / 2) * step
        extent = [x[0], x[-1], y[0], y[-1]]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

        # mêmes limites pour fusion & original
        vmin = min(img_fusion.min(), img_orig.min())
        vmax = max(img_fusion.max(), img_orig.max())

        im0 = axes[0].imshow(img_orig, origin="lower", extent=extent,
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Raw Data")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(img_fusion, origin="lower", extent=extent,
                            cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Fusion (Forward Model)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        absmax = np.max(np.abs(img_diff))
        im2 = axes[2].imshow(img_diff, origin="lower", extent=extent,
                            cmap="coolwarm", vmin=-absmax, vmax=absmax)
        axes[2].set_title("Absolute Difference (Raw − Fusion)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel("ΔRA (arcsec)")
            ax.set_ylabel("ΔDec (arcsec)")
            ax.set_aspect("equal")

        plt.suptitle(f"MIRIM — Filter {filt}", fontsize=14)
        # plt.savefig(f"/home/nmonnier/Presentations/20260123_INCLASS/comparison_MIRIM_{filt}.png", dpi=200, bbox_inches="tight")

        plt.show()



    for i, filt in enumerate(config.MIRIM.list_filters):

        # --- intégration PCE ---
        w = miri_pce[i]
        w = w / np.sum(w)
        deconv_filter = np.sum(w[:, None, None] * x_cube, axis=0)

        # --- crop ---
        img_fusion = crop_border(deconv_filter, cropy, cropx)
        img_orig   = crop_border(y_mirim[i], cropy, cropx)

        ny, nx = img_fusion.shape

        # --- axes en arcsec ---
        x = (np.arange(nx) - nx / 2) * step
        y = (np.arange(ny) - ny / 2) * step
        extent = [x[0], x[-1], y[0], y[-1]]

        # --- coupe verticale centrale ---
        xmid = nx // 2
        profile_orig   = img_orig[:, xmid]
        profile_fusion = img_fusion[:, xmid]

        # masque pour ±15 arcsec
        ymin, ymax = -11, 8
        mask_cut = (y >= ymin) & (y <= ymax)

        # --- figure : 2 lignes ---
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)

        ax_img1 = fig.add_subplot(gs[0])
        ax_img2 = fig.add_subplot(gs[1])
        # ax_prof = fig.add_subplot(gs[1, :])

        # mêmes limites pour images
        vmin = min(img_fusion.min(), img_orig.min())
        vmax = max(img_fusion.max(), img_orig.max())

        # ---- Raw ----
        im0 = ax_img1.imshow(img_orig, origin="lower", extent=extent,
                            cmap="viridis", vmin=vmin, vmax=vmax)
        # ax_img1.axvline(x=0, color="w", ls="--", lw=1.5)
        ax_img1.set_title("Raw Filter")
        plt.colorbar(im0, ax=ax_img1, fraction=0.046, pad=0.04)

        # ---- Deconv ----
        im1 = ax_img2.imshow(img_fusion, origin="lower", extent=extent,
                            cmap="viridis", vmin=vmin, vmax=vmax)
        # ax_img2.axvline(x=0, color="w", ls="--", lw=1.5)
        ax_img2.set_title("Deconv Filter (PCE integrated)")
        plt.colorbar(im1, ax=ax_img2, fraction=0.046, pad=0.04)

        

        # ax_img1.plot([0], [ymin, ymax], color="w", ls="--", lw=1.5)
        # ax_img2.plot([0], [ymin, ymax], color="w", ls="--", lw=1.5)


        # # ---- Profils ----
        # ax_prof.plot(y[mask_cut], profile_orig[mask_cut], label="Raw", lw=2, drawstyle='steps-mid')
        # ax_prof.plot(y[mask_cut], profile_fusion[mask_cut], label="Deconv", lw=2, drawstyle='steps-mid')
        # ax_prof.set_xlim(ymin, ymax)
        # ax_prof.set_xlabel("ΔDec (arcsec)")
        # ax_prof.set_ylabel("Intensity")
        # ax_prof.set_title("Vertical cut at x = 0 arcsec (±15\")")
        # ax_prof.legend()
        # ax_prof.grid(alpha=0.3)

        # labels images
        for ax in [ax_img1, ax_img2]:
            ax.set_xlabel("ΔRA (arcsec)")
            ax.set_ylabel("ΔDec (arcsec)")
            ax.set_aspect("equal")

        plt.suptitle(f"MIRIM — Filter {filt}", fontsize=14)

        # plt.savefig(
        #     f"/home/nmonnier/Presentations/20260123_INCLASS/Comparison_MIRIM_{filt}.png",
        #     dpi=200, bbox_inches="tight"
        # )
        plt.show()





    # cube_vizualisation.plot_cube(x_cube)
    cube_vizualisation.plot_on_the_fly_cube_from_maps(x_maps, templates, wavel_axis)

    cube_vizualisation.plot_mirim_comparison(y_mirim, xy_mirim, config.MIRIM.list_filters)
    




    # Plot for a specific index
    # index_wavel = 9500

    # # Crop 4 pixels on all borders
    # crop = 4
    # original_slice = cube[index_wavel, crop:-crop, crop:-crop]
    # reconstructed_slice = x_cube[index_wavel, crop:-crop, crop:-crop]
    # difference_slice = original_slice - reconstructed_slice

    # slice_idx = -1
    # # For y_mirim (2D) and xy_mirim (2D)
    # y_slice = y_mirim[slice_idx, crop:-crop, crop:-crop]
    # xy_slice = xy_mirim[slice_idx, crop:-crop, crop:-crop]
    # diff_mirim_slice = y_slice - xy_slice

    # # Compute vmin/vmax for the first line
    # vmin1 = min(original_slice.min(), reconstructed_slice.min())
    # vmax1 = max(original_slice.max(), reconstructed_slice.max())

    # # Compute vmin/vmax for the second line
    # vmin2 = min(y_slice.min(), xy_slice.min())
    # vmax2 = max(y_slice.max(), xy_slice.max())

    # # Create figure: 2 lines, 3 columns
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # # ======== FIRST ROW (Cube slices) ========

    # im0 = axes[0, 0].imshow(original_slice, vmin=vmin1, vmax=vmax1, cmap='viridis')
    # axes[0, 0].set_title('Original Slice (cropped)')
    # plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # im1 = axes[0, 1].imshow(reconstructed_slice, vmin=vmin1, vmax=vmax1, cmap='viridis')
    # axes[0, 1].set_title('Reconstructed Slice (cropped)')
    # plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # im2 = axes[0, 2].imshow(difference_slice, cmap='bwr')
    # axes[0, 2].set_title('Difference Slice (cropped)')
    # plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # # ======== SECOND ROW (y_mirim and xy_mirim) ========

    # im3 = axes[1, 0].imshow(y_slice, vmin=vmin2, vmax=vmax2, cmap='viridis')
    # axes[1, 0].set_title('y_mirim (cropped)')
    # plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # im4 = axes[1, 1].imshow(xy_slice, vmin=vmin2, vmax=vmax2, cmap='viridis')
    # axes[1, 1].set_title('xy_mirim (cropped)')
    # plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # im5 = axes[1, 2].imshow(diff_mirim_slice, cmap='bwr')
    # axes[1, 2].set_title('Difference (y - xy) (cropped)')
    # plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.show()
    
if __name__ == "__main__":
    parse_options()