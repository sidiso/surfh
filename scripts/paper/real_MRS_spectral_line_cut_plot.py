import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits

# ------------------------------------------------------
#  UTILS
# ------------------------------------------------------
def apply_mask(data, masks, wavelengths):
    """Applique un masque aux cubes de données par canal."""
    ch_limit = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 24.19]
    w_start = w_stop = 0
    for i in range(masks.shape[0]):
        w_stop = np.where(wavelengths < ch_limit[i])[0][-1] + 1
        sl = slice(w_start, w_stop)
        data[sl] *= masks[i]
        w_start = w_stop
    return data

# ------------------------------------------------------
#  PARAMÈTRES
# ------------------------------------------------------
fusion_fits = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_800_mu_5.00e+06_SD_True/res_cube.fits' 
spectral_line_idx = [935, 3773, 5070, 6611, 8426]
vertical_cut_x = 80  # colonne pour la coupe verticale

# ------------------------------------------------------
#  OUVERTURE DES DONNÉES
# ------------------------------------------------------
with fits.open(fusion_fits) as hdul:
    res_cube = hdul[0].data
    mask = hdul['MASKS'].data
    wavelength = hdul['WCS-TABLE'].data['wavelength'][0].squeeze()

masked_res_cube = apply_mask(res_cube.copy(), mask, wavelength)

# ------------------------------------------------------
#  BOUCLE SUR LES LIGNES SPECTRALES
# ------------------------------------------------------
for idx_line in spectral_line_idx:
    idx_before = idx_line - 3
    img_before = masked_res_cube[idx_before]
    img_line = masked_res_cube[idx_line]
    
    # coupe verticale (colonne fixe)
    cut_before = img_before[vertical_cut_x,:]
    cut_line = img_line[vertical_cut_x,:]
    
    # --- figure avec GridSpec ---
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0, wspace=0.3)
    
    # --- Images avec ligne verticale pour la coupe ---
    ax_img_before = fig.add_subplot(gs[0, 0])
    ax_img_line = fig.add_subplot(gs[0, 1])
    
    im1 = ax_img_before.imshow(img_before, origin='lower', cmap='viridis')
    ax_img_before.axhline(vertical_cut_x, color='red', linestyle='--', lw=1.5)
    ax_img_before.set_title(fr"Continium - $\lambda$ = {wavelength[idx_before]:.3f} μm", fontsize=12)
    fig.colorbar(im1, ax=ax_img_before, fraction=0.046, pad=0.04)
    
    im2 = ax_img_line.imshow(img_line, origin='lower', cmap='viridis')
    ax_img_line.axhline(vertical_cut_x, color='red', linestyle='--', lw=1.5)
    ax_img_line.set_title(fr"Spectral Line - $\lambda$ = {wavelength[idx_line]:.3f} μm", fontsize=12)
    fig.colorbar(im2, ax=ax_img_line, fraction=0.046, pad=0.04)
    
    # --- Coupe verticale avec double axe Y (styles A&A friendly) ---
    ax_cut = fig.add_subplot(gs[1, :])
    
    # axe Y gauche
    ax_cut.plot(np.arange(len(cut_before)), cut_before, 'k-', lw=2, label=f'Continium')
    ax_cut.set_xlabel('Pixel vertical', fontsize=12)
    ax_cut.set_ylabel(f'Intensité Index {idx_before}', color='k', fontsize=12)
    ax_cut.tick_params(axis='y', labelcolor='k', labelsize=10)
    ax_cut.tick_params(axis='x', labelsize=10)
    ax_cut.grid(True, linestyle=':', alpha=0.5)
    
    # axe Y droit
    ax2 = ax_cut.twinx()
    ax2.plot(np.arange(len(cut_line)), cut_line, 'k--', lw=2, label=fr'Spectral Line')
    ax2.set_ylabel(f'Intensité Index {idx_line}', color='k', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='k', labelsize=10)
    
    # ax_cut.set_title(f"Coupe verticale à la colonne {vertical_cut_x}", fontsize=12)
    ax_cut.legend(loc='upper right', fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)

    fig.subplots_adjust(top=0.95, bottom=0.08)
    plt.savefig(f'/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/real_MRS_cut_spectral_line_lambda_{wavelength[idx_line]:.1f}.png', dpi=300)
    plt.savefig(f'/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/real_MRS_cut_spectral_line_lambda_{wavelength[idx_line]:.1f}.pdf')


    plt.show()
