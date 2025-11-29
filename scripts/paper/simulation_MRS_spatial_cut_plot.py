import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt


def apply_mask(data, masks, wavelengths):
    """Apply a mask to the data cube.

    Parameters
    ----------
    mask : np.ndarray
        A boolean array with the same spatial dimensions as the data cube.
        True values indicate valid data points, while False values indicate masked points.
    """
    ch_limit = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 24.19]
    # Apply the mask to each wavelength slice
    w_start = w_stop = 0
    for i in range(masks.shape[0]):
        w_stop = np.where(wavelengths < ch_limit[i])[0][-1] +1
        slice_mask = slice(w_start, w_stop)
        data[slice_mask,:,:] = data[slice_mask,:,:]*masks[i]
        w_start = w_stop

    return data

# Load data
ref_cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
templates = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_templates.npy')

# for i in range(templates.shape[0]):
#     print(f'Wavelength where template is not zero for template {i}: {np.where(templates[i,:]!=0)[0]}')
spectral_line_idx = [937, 1740, 2678, 3778, 5077, 6619, 7537, 8435]

# Load results data
with fits.open('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Results/lcg_MC_11_MO_4_Temp_13_nit_200_mu_5.00e+05_SD_True/res_cube.fits') as hdul:
    res_cube = hdul[0].data
    mask = hdul['MASKS'].data
    wavelength = hdul['WCS-TABLE'].data['wavelength'][0]
    wavelength = wavelength.squeeze() # Because shape can be (N,1) or (1,N)

masked_res_cube = apply_mask(res_cube, mask, wavelength)
masked_ref_cube = apply_mask(ref_cube, mask, wavelength)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

wavelengths_to_plot = [6.2, 7.7, 8.6, 11.3]
indices_to_plot = [np.abs(wavelength - wl).argmin() for wl in wavelengths_to_plot]

y_cut = 60  # coupe horizontale

# Figure 2x2, mais chaque cellule = (image + plot) dans un axe composite
fig, axs = plt.subplots(2, 2, figsize=(11, 9))

for ax_main, idx, wl in zip(axs.flatten(), indices_to_plot, wavelengths_to_plot):

    # On découpe l'axe principal en 2 sous-axes verticaux
    divider = make_axes_locatable(ax_main)
    ax_img = divider.append_axes("top", size="65%", pad=0)
    ax_cut = divider.append_axes("bottom", size="35%", pad=0.4)

    # ====================
    #   1) IMAGE 2D
    # ====================
    slice_res = masked_res_cube[idx]

    im = ax_img.imshow(slice_res, origin="lower", cmap="inferno")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"Slice @ {wl:.1f} μm")

    # Ligne de coupe superposée
    ax_img.axhline(y_cut, color='white', lw=1.4, ls='--', alpha=0.9)
    ax_img.axhline(y_cut, color='red', lw=1.0, ls='--', alpha=0.9)

    # Colorbar fine collée à droite
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)

    # ====================
    #   2) COUPE SPATIALE
    # ====================
    spatial_cut_ref = masked_ref_cube[idx, :, y_cut]
    spatial_cut_res = masked_res_cube[idx, :, y_cut]

    ax_cut.plot(spatial_cut_ref, label="Reference", color="black", lw=1.4)
    ax_cut.plot(spatial_cut_res, label="Reconstructed", color="red", lw=1.0, ls="--")

    ax_cut.set_xlim(0, slice_res.shape[1])
    ax_cut.set_ylabel("Flux")
    ax_cut.set_xlabel("X Pixel")
    ax_cut.grid(True, ls="--", alpha=0.4)

    if idx == indices_to_plot[0]:
        ax_cut.legend(fontsize=8)

    # on enlève l'axe principal (qui est juste un conteneur)
    ax_main.set_visible(False)

plt.tight_layout()
plt.show()
