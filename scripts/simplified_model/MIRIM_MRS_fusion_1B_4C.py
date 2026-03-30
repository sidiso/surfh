import numpy as np 
import os

from surfh.Models.simplifiedMIRIM import Mirim_Model_For_Fusion
from surfh.Models.simplifiedMRS import Spectro_Model_3
from surfh.Algorithm.criteria_simplified import QuadCriterion2
from surfh.Models.lmm import LinearMixingModel
from surfh.Vizualisation.cube_vizualisation import plot_cube
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits
from surfh.Models import wavelength_mrs
from surfh.ToolsDir.matrix_op import lmm_maps2cube


import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import trapezoid
from matplotlib.colors import LogNorm  #

import time
import pathlib

fusion_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'

pce_mirim_dir  = fusion_dir + 'Templates/PCE_MIRIM/Simplified/'
wavel_axis = np.load(fusion_dir + 'Templates/wavelength/simplified_model_wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy')


psfs = np.load(fusion_dir + 'PSF/simplified_model_wavel_psfs_pixscale0.1_npix_212x36_chan_1ABC_2ABC_3ABC_4ABC.npy')#[:len(wavel_axis),:,:]
# psfs = psfs[is_in_A,:,:] # keep only psfs corresponding to wavel_axis after removing spectral lines

# PCE alredy in the right shape (without spectral lines)
list_pce = []
for file in sorted(os.listdir(pce_mirim_dir)) :
    print(f'Load PCE file for from file {file} ')
    list_pce.append(np.load(pce_mirim_dir+file)[0])
pce = np.array(list_pce)


# Select MRS bands and correspondind indexes 
mrs_lim_band = ['ch1b', 'ch4c']
min_wavel = wavelength_mrs.get_mrs_wavelength(mrs_lim_band[0])[0]
max_wavel = wavelength_mrs.get_mrs_wavelength(mrs_lim_band[1])[-1]
start_idx = np.argmin(np.abs(wavel_axis - min_wavel))
end_idx = np.argmin(np.abs(wavel_axis - max_wavel))

spectral_cut_before = 50
spectral_cut = 200
# Select only wavelengths and psfs in the MRS bands
wavel_axis = wavel_axis[start_idx+spectral_cut_before:end_idx-spectral_cut]
psfs = psfs[start_idx+spectral_cut_before:end_idx-spectral_cut, :, :]
pce_mirim = pce[1:, start_idx+spectral_cut_before:end_idx-spectral_cut]
pce_spectro = np.ones_like(wavel_axis)


templates = np.load(fusion_dir + 'Templates/NMF/full_scan_simplified_ch1b_to_ch4c_12_nmf_components_and_8_spectral_lines.npy')
templates= templates[:,spectral_cut_before:]


# pce_mirim = pce[:-1, :len(wavel_axis)]
print("wavel_axis mirim shape = ", wavel_axis.shape)
print("PCE mirim shape = ", pce_mirim.shape)
print("PSFs shape = ", psfs.shape)
print("Templates shape = ", templates.shape)


# raise ValueError("Stop here to check PCE shapes")




imshape = (212,36)

L_SS = 1 # Sous-échantillonnage spectrale pour la simulation

print("Shapes before SS: ")
print("psfs shape = ", psfs.shape)
print("wavel_axis shape = ", wavel_axis.shape)
print("templates shape = ", templates.shape)
print("PCE Shape = ", pce_mirim.shape)


psfs_SS = psfs[::L_SS,:,:]  # sous-échantillonnage des psfs pour la simulation
wavel_axis_SS = wavel_axis[::L_SS]  # sous-échantillonnage des longueurs d'onde pour la simulation
templates_SS = templates[:,::L_SS]  # sous-échantillonnage des templates pour la simulation
pce_spectro_SS = pce_spectro[::L_SS]  # sous-échantillonnage du pce pour la simulation
pce_mirim_SS = pce_mirim[:,::L_SS]  # sous-échantillonnage du pce pour la simulation


print("Shapes after SS: ")  
print("psfs_SS shape = ", psfs_SS.shape)
print("wavel_axis_SS shape = ", wavel_axis_SS.shape)
print("templates_SS shape = ", templates_SS.shape)
print("pce_spectro_SS shape = ", pce_spectro_SS.shape)
print("pce_mirim_SS shape = ", pce_mirim_SS.shape)
print("Wavelength range after SS: ", wavel_axis_SS.min(), " to ", wavel_axis_SS.max())


decim = 2
# facteurs de décimations spatiales pour le modèle spectro
di = decim
dj = decim

# reshape mirim model for the fusion model
start = time.time()
# h_int = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/hint_124pix_13templates_SS10.npy')
h_int = None
mirim_model_for_fusion = Mirim_Model_For_Fusion(
    psfs_SS, pce_mirim_SS, wavel_axis_SS, templates_SS, imshape, di, dj, precomputed_H_int = h_int, low_mem=True
)
# np.save('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/hint_124pix_13templates_SS10.npy', mirim_model_for_fusion.H_int)
end = time.time()
print(f"Time to create mirim model for fusion: {end - start} seconds")


# creation of spectro model from scratch
start = time.time()
spectro_model = Spectro_Model_3(
    psfs_SS, pce_spectro_SS, di, dj, wavel_axis_SS, templates_SS, imshape
)
end = time.time()
print(f"Time to create spectro model: {end - start} seconds")

# Load MIRIM data
start = time.time()
y_mirim_list = []
for mirim_data_file in sorted(os.listdir('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Corrected_MIRIM/simplified')):
    print(f'Load MIRIM data from file {mirim_data_file} ')
    y_mirim_list.append(np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Corrected_MIRIM/simplified/' + mirim_data_file))
y_mirim = np.array(y_mirim_list)
print("y_mirim shape = ", y_mirim.shape)
y_mirim = y_mirim[1:]
print("y_mirim shape after removing last channel = ", y_mirim.shape)
end = time.time()
print(f"Time to compute y_mirim: {end - start} seconds")


# Load and preprocess MRS data
start = time.time()
from astropy.io import fits
raw_y_spectro = np.load('/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/Channels/rotated_merged_foi_cube.npy')
# raw_y_spectro = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Filtered_raw_cube.fits')
# hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Filtered_raw_cube.fits')
# raw_y_spectro = hdul[0].data
# raw_y_spectro = raw_y_spectro[spectral_cut_before:,:,:]

end = time.time()
print(f"Time to compute y_spectro: {end - start} seconds")
print("y_spectro shape = ", raw_y_spectro.shape)
raw_y_spectro = raw_y_spectro[start_idx+spectral_cut_before:end_idx-spectral_cut,:,:]
raw_y_spectro = raw_y_spectro[::L_SS, :, :]

# N, H, W = raw_y_spectro.shape
# H4, W4 = (H//decim)*decim, (W//4)*decim
# tab2 = raw_y_spectro[:, :H4, :W4]



"""
Test rapport spectro imageur
"""
# plt.figure()
# for i, filt in enumerate(['F770W', 'F1000W', 'F1130W', 'F1280W','F1500W', 'F1800W', 'F2100W', 'F2550W']):
#     print(pce_mirim_SS.shape)
#     plt.plot(wavel_axis_SS, pce_mirim_SS[i])
# plt.show()

corrected_y_mirim = y_mirim.copy()
for i, filt in enumerate(['F770W', 'F1000W', 'F1130W', 'F1280W',
                          'F1500W', 'F1800W', 'F2100W', 'F2550W']):

    # Integrate MRS data over PCE of the filter
    pce_filt = pce_mirim_SS[i]
    delta_lambda = np.gradient(wavel_axis_SS)

    pce_norms = np.sum(pce_filt)
    # y_spectro_filt = np.sum(
    #     raw_y_spectro * pce_filt[:, None, None] * delta_lambda[:, None, None],
    #     axis=0
    # )
    print(raw_y_spectro.shape, pce_filt.shape, wavel_axis_SS.shape)


    pce_norm = trapezoid(pce_filt, x=wavel_axis_SS)
    y_spectro_filt = trapezoid(raw_y_spectro*pce_filt[:,None,None], x=wavel_axis_SS, axis=0)/pce_norm
    print(y_spectro_filt.shape)


    y_imager = y_mirim[i]

    # Flatten
    x = y_imager.flatten()
    y = y_spectro_filt.flatten()

    # Remove NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

#     # ---------- Fit optimal: y = a x + b ----------
    a, b = np.polyfit(x, y, 1)
    corrected_y_mirim[i] *= a
#     y_imager2 = corrected_y_mirim[i]
#     x2 = y_imager2.flatten()
#     y2 = y_spectro_filt.flatten()

#     plt.figure()
#     plt.imshow(corrected_y_mirim[i] - y_spectro_filt, origin="lower")
#     plt.colorbar()
#     plt.title(f"Ymirim - Ymrs filt = {filt}")

#     a2, b2 = np.polyfit(x2, y2, 1)

#     # corrected_y_mirim 

#     # Lines for plotting
#     x_line = np.linspace(x.min(), x.max(), 200)
#     y_fit = a * x_line + b          # best fit

#     y_fit2 = a2*x_line +b2


#     y_unity = x_line.copy()                # slope = 1, through origin
#     y_unity -= y_unity[0]
#     y_unity += y_fit[0]
#     # y_unity += (y_unity[0] - y_fit[0])  # shift to pass through first point

#     # ---------- Plot ----------
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(x, y, '.', alpha=0.1, label='Pixels')
#     # plt.plot(x_line, y_fit, 'r-', linewidth=2,
#     #          label=f'Fit: y = {a:.3f} x + {b:.3f}')
#     # plt.plot(x_line, y_unity, 'k--', linewidth=2,
#     #          label='Unity slope: y = x')

#     # plt.xlabel('MIRIM Flux')
#     # plt.ylabel(f'MRS Flux integrated over {filt}')
#     # plt.legend()

#     # plt.figure(figsize=(10, 6))
#     # plt.plot(x2, y2, '.', alpha=0.1, label='Pixels')
#     # plt.plot(x_line, y_fit2, 'r-', linewidth=2,
#     #          label=f'Fit: y = {a2:.3f} x + {b2:.3f}')

#     # plt.xlabel('Corrected MIRIM Flux')
#     # plt.ylabel(f'MRS Flux integrated over {filt}')
#     # plt.legend()


#     plt.show()

# TODEL 
y_mirim = corrected_y_mirim.copy()

# y_spectro = tab2.reshape(N, H4//4, 4, W4//4, 4).mean(axis=(2, 4))
from skimage.measure import block_reduce
import numpy as np

y_spectro = block_reduce(raw_y_spectro, block_size=(1, decim, decim), func=np.sum)

print("y_spectro shape after decimation: ", y_spectro.shape)

weight_mirim = np.sum(y_mirim)
weight_spectro = np.sum(y_spectro)

mu_imager = 1
mu_spectro = 1 * (weight_mirim / weight_spectro)

list_mu_reg = [1, 1e1, 7e1, 1e2, 5e2, 1e3, 5e3, 1e4]
list_mu_spectro = [mu_spectro * factor for factor in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]]

mu_spectro = list_mu_spectro[4]
name = "MuMiddle"

print(f"mu_spectro = {mu_spectro}")
mu_reg = 1e1

""""""""""""""""""""""""""""""""""""""""""""""""""
y_mrs_mean_spectrum = np.mean(raw_y_spectro, axis=(1,2))
list_mu_reg = [1, 1e1, 2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 1e2, 3e2, 5e2, 1e3, 5e3, 1e4]
list_mu_spectro = [mu_spectro * factor for factor in [0.001,0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 50]]
# map_res_spectro = np.zeros((len(list_mu_reg), len(list_mu_spectro)))
# map_res_imager = np.zeros((len(list_mu_reg), len(list_mu_spectro)))
# for i, mu_reg in enumerate(list_mu_reg):
#     array_mu_reg = np.zeros((templates_SS.shape[0])) + mu_reg
#     for k in range(12, templates_SS.shape[0]):
#         array_mu_reg[k] = 1e3

#     for j, mu_spectro in enumerate(list_mu_spectro):
#         print(f'Running fusion with mu_reg = {mu_reg} and mu_spectro = {mu_spectro}')
        
#         quadcriterion = QuadCriterion2(
#             mu_imager,
#             y_mirim,
#             mirim_model_for_fusion,
#             mu_spectro,
#             y_spectro,
#             spectro_model,
#             array_mu_reg,
#             printing = False,
#             gradient = "separated"
#         )

#         start = time.time()
#         quadcrit_rec_maps = quadcriterion.run_expsol()
#         end = time.time()
#         print(f"Time to run exp sol: {end - start} seconds")

#         res_cube = lmm_maps2cube(quadcrit_rec_maps, templates_SS)
#         xy_mirim = mirim_model_for_fusion.forward(quadcrit_rec_maps)

#         # Save res spectro
#         res_spectro = np.mean(np.abs((np.mean(res_cube, axis=(1,2))- y_mrs_mean_spectrum)/y_mrs_mean_spectrum))

#         # Save mean values for mirim without borders of 2 pixels
#         res_imager = np.mean(np.abs((y_mirim[:,2:-2,2:-2]- xy_mirim[:,2:-2,2:-2])/y_mirim[:,2:-2,2:-2]))

#         map_res_spectro[i,j] = res_spectro
#         map_res_imager[i,j] = res_imager

# plt.figure()
# plt.imshow(map_res_spectro, origin='lower', cmap='viridis', aspect='auto')
# plt.colorbar(label='Mean absolute relative error spectrum')
# plt.xlabel('mu_spectro')
# plt.ylabel('mu_reg')
# plt.title('Mean absolute relative error spectrum vs mu_spectro and mu_reg')
# plt.savefig('/home/nmonnier/Presentations/20260216_L2S/Mean_absolute_relative_error_spectrum_vs_mu_spectro_and_mu_reg.png', dpi=300)
# plt.figure()
# plt.imshow(map_res_imager, origin='lower', cmap='viridis', aspect='auto')
# plt.colorbar(label='Mean absolute relative error imager')
# plt.xlabel('mu_spectro')
# plt.ylabel('mu_reg')
# plt.title('Mean absolute relative error imager vs mu_spectro and mu_reg')
# plt.savefig('/home/nmonnier/Presentations/20260216_L2S/Mean_absolute_relative_error_imager_vs_mu_spectro_and_mu_reg.png', dpi=300)
# plt.show()


# raise SystemExit
""""""""""""""""""""""""""""""""""""""""""""""""""
start = time.time()
array_mu_reg = np.zeros((templates_SS.shape[0])) + list_mu_reg[3]   
# Set reg for element 12 to the end
for k in range(12, templates_SS.shape[0]):
    array_mu_reg[k] = 1e3
quadcriterion = QuadCriterion2(
    mu_imager,
    y_mirim,
    mirim_model_for_fusion,
    list_mu_spectro[10],
    y_spectro,
    spectro_model,
    array_mu_reg,#list_mu_reg[3],
    printing = False,
    gradient = "separated"
)
end = time.time()
print(f"Time to create quadcriterion: {end - start} seconds")

start = time.time()
quadcrit_rec_maps = quadcriterion.run_expsol()
end = time.time()
print(f"Time to run exp sol: {end - start} seconds")
print(quadcrit_rec_maps.shape)
# Linear_Mixture_Model.mapsToCube(quadcrit_rec_maps, templates_SS)

from surfh.ToolsDir.matrix_op import lmm_maps2cube
res_cube = lmm_maps2cube(quadcrit_rec_maps, templates_SS)

# Save results
result_dir = f'MIRIM_MRS_EXP_Simplified_mu_{mu_reg}_muImg_{mu_imager}_muMRS_{mu_spectro:.2f}/'

path = pathlib.Path('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/' + result_dir)
path.mkdir(parents=True, exist_ok=True)
metadata = {"WAVELENGTH": wavel_axis_SS}
save_numpy_to_fits(res_cube, metadata, path/'Reconstructed_cube.fits', masks=None)

# true_cube = LinearMixingModel.mapsToCube(quadcrit_rec_maps, templates_SS)
# plot_cube(true_cube, wavel_axis_SS, title='Reconstructed Cube')


hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Filtered_raw_cube.fits')
raw_y_spectro = hdul[0].data
raw_y_spectro = raw_y_spectro[spectral_cut_before:,:,:]




y_mrs_mean_spectrum = np.nanmean(raw_y_spectro[:,2:-2,2:-2], axis=(1,2))
xy_mirim = mirim_model_for_fusion.forward(quadcrit_rec_maps)

# REmove spectral line of mean spectrum based on templates from index 12 to the end
# Because these templates are 0 but at the specific wavelengths, we make set the fluw equal to the mean between the wavelengths before and after the spectral line
for k in range(12, templates_SS.shape[0]):
    # print(f"Removing spectral line at wavelength {wavel_axis_SS[np.nonzero(templates_SS[k])[0]]:.2f} microns")
    indices = np.nonzero(templates_SS[k])[0]
    y_mrs_mean_spectrum[indices] = (y_mrs_mean_spectrum[indices[0]-1]+y_mrs_mean_spectrum[indices[-1]+1])/2


# Flip all data
for i in range(xy_mirim.shape[0]):
    xy_mirim[i] = np.fliplr(xy_mirim[i])
    y_mirim[i] = np.fliplr(y_mirim[i])
for i in range(res_cube.shape[0]):
    res_cube[i] = np.fliplr(res_cube[i])
    raw_y_spectro[i] = np.fliplr(raw_y_spectro[i])





mean_rec = np.mean(res_cube[:,2:-2,2:-2], axis=(1, 2))
for k in range(12, templates_SS.shape[0]):
    # print(f"Removing spectral line at wavelength {wavel_axis_SS[np.nonzero(templates_SS[k])[0]]:.2f} microns")
    indices = np.nonzero(templates_SS[k])[0]
    mean_rec[indices] = (mean_rec[indices[0]-1]+mean_rec[indices[-1]+1])/2

mean_rec = mean_rec[:-200]
y_mrs_mean_spectrum = y_mrs_mean_spectrum[:-200]
rel_err = (mean_rec - y_mrs_mean_spectrum) / y_mrs_mean_spectrum

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)

# ---- Spectres ----
axes[0].plot(wavel_axis_SS[:-200], mean_rec, label=r"Fused", lw=2, linestyle="dotted")
axes[0].plot(wavel_axis_SS[:-200], y_mrs_mean_spectrum, label=r"MRS", lw=2, linestyle="--")
axes[0].set_ylabel("Flux (MJy/sr)", fontsize=14)
# axes[0].set_title("Mean MRS Spectrum", fontsize=14)
axes[0].legend(fontsize=13)
axes[0].grid(True, alpha=0.4)

# ---- Erreur relative ----
axes[1].plot(wavel_axis_SS[:-200], rel_err, 
             lw=2, 
             label=r"$\frac{\bar{\boldsymbol{y}}_{\mathrm{h}} - \bar{\hat{\boldsymbol{x}}}}{\bar{\hat{\boldsymbol{x}}}}$"
            )
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel(r"Wavelength ($\mu$m)", fontsize=14)
axes[1].set_ylabel("Relative Error", fontsize=14)
# axes[1].set_title("Relative Spectral Error", fontsize=14)
axes[1].legend(fontsize=13)
axes[1].grid(True, alpha=0.4)

# ---- ticks plus gros ----
for ax in axes:
    ax.tick_params(axis="both", labelsize=13)

print("Mean absolute relative error spectrum:",
      np.mean(np.abs(rel_err)))
# plt.savefig(
#             f"/home/nmonnier/Presentations/20260225_AMIS/Fusion_MRS_MIRIM_{name}_MeanFlux.png",
#             dpi=300, bbox_inches="tight"
#         )

plt.show()



def crop2(img, n=2):
    return img[n:-n, n:-n]
filters_name = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']





crop = 3
pixscale = 0.1
filter_wavelength = 7.7
idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))

pce_filt = pce_mirim_SS[0]
pce_norm = trapezoid(pce_filt, x=wavel_axis_SS)
y_spectro_filt = trapezoid(raw_y_spectro*pce_filt[:,None,None], x=wavel_axis_SS, axis=0)/pce_norm






# --------------------------------------------------
# Plot FILTER
# --------------------------------------------------
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_filter_comparison(
    filter_wavelength, mirim_idx, mirim_label,
    wavel_axis_SS, raw_y_spectro, y_mirim, res_cube, xy_mirim,
    pixscale, crop
):
    idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))

    img0 = np.rot90(crop2(raw_y_spectro[idx_filter], crop))
    img1 = np.rot90(crop2(y_mirim[mirim_idx], crop))
    img2 = np.rot90(crop2(res_cube[idx_filter], crop))
    xy_img = np.rot90(crop2(xy_mirim[mirim_idx], crop))
    residual = (xy_img - img1) / img1

    imgs = [img0, img1, img2]
    titles = [
        fr"MRS Raw $\boldsymbol{{y}}_{{h}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",
        fr"MIRIM Filter $\boldsymbol{{y}}_{{m}}$ — {mirim_label}",
        fr"Fused $\widehat{{\boldsymbol{{x}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",
    ]

    ny, nx = img0.shape
    x = (np.arange(nx) - nx / 2) * pixscale
    y = (np.arange(ny) - ny / 2) * pixscale
    extent = [x[0], x[-1], y[0], y[-1]]

    vmin = min(img.min() for img in imgs)
    vmax = max(img.max() for img in imgs)
    rmax = np.max(np.abs(residual))

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)

    fig.set_constrained_layout_pads(
        w_pad=0.0,
        h_pad=0.02,
        wspace=0.0005   # ← réduit l'espace entre images et colorbar
    )

    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    for ax, img, title in zip(axes[:3], imgs, titles):
        im = ax.imshow(img, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("ΔRA (arcsec)")
        ax.set_ylabel("ΔDec (arcsec)")
        ax.set_aspect("equal")


    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip(axes, labels):
        ax.text(
            -0.1, 0.5,        # x, y en coordonnées relatives à l'axe
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="normal",
            va="top", ha="right"
        )


    # Après avoir tracé les 3 images, récupérer les positions des axes
    pos0 = axes[0].get_position()  # axe du haut
    pos2 = axes[2].get_position()  # axe du bas

    # Placer la colorbar à droite, alignée sur les 3 axes
    cax = fig.add_axes([
        pos0.x1 - 0.05,   # x : juste à droite des images
        pos2.y0+0.09,           # y : bas du 3ème axe
        0.025,             # largeur
        pos0.y1 - pos2.y0  # hauteur : du bas du 3ème au haut du 1er
    ])

    fig.colorbar(im, cax=cax).set_label("Flux (MJy/sr)", fontweight="bold")    



    norm = TwoSlopeNorm(vcenter=0, vmin=-rmax, vmax=rmax)
    imr = axes[3].imshow(residual, origin="lower", extent=extent, cmap="RdBu_r", norm=norm)
    axes[3].set_title(fr"Relative residual — $(\boldsymbol{{H}}_{{m}}\widehat{{\boldsymbol{{x}}}} - \boldsymbol{{y}}_{{m}})/\boldsymbol{{y}}_{{m}}$", fontweight="bold")
    axes[3].set_xlabel("ΔRA (arcsec)")
    axes[3].set_ylabel("ΔDec (arcsec)")
    axes[3].set_aspect("equal")

    fig.colorbar(imr, ax=axes[3], orientation="horizontal", pad=0.25).set_label("Residual Intensity", fontweight="bold")

    # plt.savefig(
    #     f"/home/nmonnier/Presentations/20260123_INCLASS/Fusion_MRS_MIRIM_{mirim_label}.png",
    #     dpi=300, bbox_inches="tight"
    # )

    plt.show()


filters = [
    (10.0,  1, "F1000W"),
    (15.0,  4, "F1500W"),
    (18.0,  5, "F1800W"),
    (21.0,  6, "F2100W"),
    (25.5,  7, "F2550W"),
]

for wavelength, mirim_idx, label in filters:
    plot_filter_comparison(
        wavelength, mirim_idx, label,
        wavel_axis_SS, raw_y_spectro, y_mirim, res_cube, xy_mirim,
        pixscale, crop
    )




############################################################""
## Spatial cut continium
##############################################################
pixscale = 0.1
crop_tb = 75
crop_lr = 3
pixel_cut = 12-4


wavelength_cuts = [7.7, 10.0, 11.3, 12.8, 15.0, 18.0, 21.0, 25.5]
filters = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']

for i, wavelength_cut in enumerate(wavelength_cuts):
    pce_filt = pce_mirim_SS[i]
    pce_norm = trapezoid(pce_filt, x=wavel_axis_SS)
    res_cube_filt = trapezoid(res_cube * pce_filt[:, None, None], x=wavel_axis_SS, axis=0) / pce_norm
    mirim_filter = y_mirim[i]

    idx = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
    img_rec = res_cube[idx]
    img_mrs = raw_y_spectro[idx]

    # ---- crop ----
    img_rec_c       = img_rec[crop_tb:-crop_tb, crop_lr:-crop_lr]
    img_mrs_c       = img_mrs[crop_tb:-crop_tb, crop_lr:-crop_lr]
    mirim_filter_c  = mirim_filter[crop_tb:-crop_tb, crop_lr:-crop_lr]
    res_cube_filt_c = res_cube_filt[crop_tb:-crop_tb, crop_lr:-crop_lr]

    ny, nx = img_rec_c.shape

    # ---- axes en arcsec ----
    x = (np.arange(nx) - nx / 2) * pixscale
    y = (np.arange(ny) - ny / 2) * pixscale
    extent = [x[0], x[-1], y[0], y[-1]]

    # ---- profils ----
    prof_rec          = img_rec_c[:, pixel_cut]
    prof_mrs          = img_mrs_c[:, pixel_cut]
    prof_mirim        = mirim_filter_c[:, pixel_cut]
    prof_spectro_filt = res_cube_filt_c[:, pixel_cut]

    # ---- layout : 2 blocs côte à côte ----
    # Chaque bloc = 2 colonnes d'images + 1 ligne de profil
    # Grille globale : 2 lignes × 4 colonnes
    fig = plt.figure(figsize=(20, 9))
    # fig.suptitle(
    #     rf"Filter {filters[i]} — $\lambda_\mathrm{{cut}}={wavelength_cut}\,\mu\mathrm{{m}}$  "
    #     rf"($\lambda_\mathrm{{actual}}={wavel_axis_SS[idx]:.2f}\,\mu\mathrm{{m}}$)",
    #     fontsize=14, fontweight="bold"
    # )

    # Séparateur visuel entre les deux blocs via un espace de colonne
    # gs : 2 lignes, 5 colonnes (col 0-1 = bloc 1, col 2 = spacer, col 3-4 = bloc 2)
    gs = fig.add_gridspec(
        2, 5,
        height_ratios=[1, 0.6],
        width_ratios=[1, 1, 0.08, 1, 1],
        hspace=0.45, wspace=0.35
    )

    # ---------- BLOC 1 : MRS  vs  Fused ----------
    ax_mrs = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_prof1 = fig.add_subplot(gs[1, 0:2])

    vmin1 = min(img_mrs_c.min(), img_rec_c.min())
    vmax1 = max(img_mrs_c.max(), img_rec_c.max())

    im_mrs = ax_mrs.imshow(img_mrs_c, origin="lower", extent=extent,
                           cmap="viridis", vmin=vmin1, vmax=vmax1)
    ax_mrs.plot([x[pixel_cut]] * 2, [y[0], y[-1]], "r--", lw=1.5)
    ax_mrs.set_title(r"MRS $\mathbf{y}_{h}$")
    plt.colorbar(im_mrs, ax=ax_mrs, fraction=0.046, pad=0.04)

    im_rec = ax_rec.imshow(img_rec_c, origin="lower", extent=extent,
                           cmap="viridis", vmin=vmin1, vmax=vmax1)
    ax_rec.plot([x[pixel_cut]] * 2, [y[0], y[-1]], "r--", lw=1.5)
    ax_rec.set_title(r"Fused $\hat{\mathbf{x}}$ ")
    plt.colorbar(im_rec, ax=ax_rec, fraction=0.046, pad=0.04)

    for ax in [ax_mrs, ax_rec]:
        ax.set_xlabel(r"$\Delta$RA ($^{\prime\prime}$)")
        ax.set_ylabel(r"$\Delta$Dec ($^{\prime\prime}$)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    ax_prof1.plot(y, prof_mrs, label="MRS",          lw=2, drawstyle="steps-mid")
    ax_prof1.plot(y, prof_rec, label="Fused", lw=2, drawstyle="steps-mid")
    ax_prof1.set_xlabel(r"$\Delta$Dec ($^{\prime\prime}$)", fontsize=11)
    ax_prof1.set_ylabel("Flux (MJy/sr)", fontsize=11)
    ax_prof1.set_title(
        rf"Vertical cut at {x[pixel_cut]:.2f}''$ — $\lambda={wavel_axis_SS[idx]:.2f}\,\mu$m",
        fontsize=13
    )
    ax_prof1.grid(True, alpha=0.4)
    ax_prof1.legend(fontsize=10)

    # ---------- BLOC 2 : MIRIM filter  vs  y_spectro_filt ----------
    ax_mirim  = fig.add_subplot(gs[0, 3])
    ax_sfilt  = fig.add_subplot(gs[0, 4])
    ax_prof2  = fig.add_subplot(gs[1, 3:5])

    vmin2 = min(mirim_filter_c.min(), res_cube_filt_c.min())
    vmax2 = max(mirim_filter_c.max(), res_cube_filt_c.max())

    im_mirim = ax_mirim.imshow(mirim_filter_c, origin="lower", extent=extent,
                               cmap="viridis", vmin=vmin2, vmax=vmax2)
    ax_mirim.plot([x[pixel_cut]] * 2, [y[0], y[-1]], "r--", lw=1.5)
    ax_mirim.set_title(rf"MIRIM $\boldsymbol{{y}}_{{\mathrm{{m}}}}$ - {filters[i]}")
    plt.colorbar(im_mirim, ax=ax_mirim, fraction=0.046, pad=0.04)

    im_sfilt = ax_sfilt.imshow(res_cube_filt_c, origin="lower", extent=extent,
                               cmap="viridis", vmin=vmin2, vmax=vmax2)
    ax_sfilt.plot([x[pixel_cut]] * 2, [y[0], y[-1]], "r--", lw=1.5)
    ax_sfilt.set_title(rf"$\boldsymbol{{W}}_{{m}}\hat{{\boldsymbol{{x}}}}$ - {filters[i]}")
    plt.colorbar(im_sfilt, ax=ax_sfilt, fraction=0.046, pad=0.04)

    for ax in [ax_mirim, ax_sfilt]:
        ax.set_xlabel(r"$\Delta$RA ($^{\prime\prime}$)")
        ax.set_ylabel(r"$\Delta$Dec ($^{\prime\prime}$)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    ax_prof2.plot(y, prof_mirim,        label=rf"MIRIM", lw=2, drawstyle="steps-mid")
    ax_prof2.plot(y, prof_spectro_filt, label=r"$\boldsymbol{W}_{m}\hat{\boldsymbol{x}}$", lw=2, drawstyle="steps-mid")
    ax_prof2.set_xlabel(r"$\Delta$Dec ($^{\prime\prime}$)", fontsize=11)
    ax_prof2.set_ylabel("Flux (MJy/sr)", fontsize=11)
    ax_prof2.set_title(
        rf"Vertical cut at ${x[pixel_cut]:.2f}'' - Filter {filters[i]}",
        fontsize=13
    )
    ax_prof2.grid(True, alpha=0.4)
    ax_prof2.legend(fontsize=10)

    # plt.savefig(
    #     f"/home/nmonnier/Presentations/20260123_INCLASS/Cut_{pixel_cut}_Fusion_MRS_MIRIM_{filters[i]}.png",
    #     dpi=300, bbox_inches="tight"
    # )
    plt.show()




######################################

#######################################
for filter in range(y_mirim.shape[0]):
    print(f'Filter {filters_name[filter]}, Flux ratio (xy_mirim/y_mirim), {np.sum(xy_mirim[filter][50:150, 3:-3])/np.sum(y_mirim[filter][50:150, 3:-3])}')
    print(f"Filter {filters_name[filter]}, Frobenius relative error, {np.linalg.norm(y_mirim[filter][50:150, 3:-3] - xy_mirim[filter][50:150, 3:-3], 'fro')/np.linalg.norm(y_mirim[filter][50:150, 3:-3], 'fro')}")
    print(f'Filter {filters_name[filter]}, Flux ratio (xy_mirim/y_mirim), {np.sum(xy_mirim[filter][3:-3, 3:-3])/np.sum(y_mirim[filter][3:-3, 3:-3])}')
    print(f"Filter {filters_name[filter]}, Frobenius relative error, {np.linalg.norm(y_mirim[filter][3:-3, 3:-3] - xy_mirim[filter][3:-3, 3:-3], 'fro')/np.linalg.norm(y_mirim[filter][3:-3, 3:-3], 'fro')}")






pixscale = 0.1
crop_tb = 75
crop_lr = 3
pixel_cut = 12-4


hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Filtered_raw_cube.fits')
raw_y_spectro = hdul[0].data
# raw_y_spectro = raw_y_spectro[spectral_cut_before:,:,:]
for i in range(raw_y_spectro.shape[0]):
    raw_y_spectro[i] = np.fliplr(raw_y_spectro[i])
wavel_axis_SS = hdul['WCS-TABLE'].data['wavelength'][0].squeeze()


wavelength_spectral_line_cuts = [6.108, 6.909, 8.025, 9.664, 12.278, 17.032]
i = 0
for wavelength_cut in wavelength_spectral_line_cuts:



    idx = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))

    img_rec = raw_y_spectro[idx]
    img_cont = raw_y_spectro[idx-7]

    # ---- crop top/bottom ----
    img_rec_c = img_rec[crop_tb:-crop_tb, crop_lr:-crop_lr]
    img_mrs_c = img_cont[crop_tb:-crop_tb, crop_lr:-crop_lr]

    ny, nx = img_rec_c.shape

    # ---- axes en arcsec ----
    x = (np.arange(nx) - nx / 2) * pixscale
    y = (np.arange(ny) - ny / 2) * pixscale
    extent = [x[0], x[-1], y[0], y[-1]]

    # ---- profils ----
    prof_rec = img_rec_c[:, pixel_cut]
    prof_mrs = img_mrs_c[:, pixel_cut]

    # ---- layout ----
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax_mrs = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    # ---- images (même échelle) ----
    vmin = min(img_mrs_c.min(), img_rec_c.min())
    vmax = max(img_mrs_c.max(), img_rec_c.max())

    im_mrs = ax_mrs.imshow(img_mrs_c, origin="lower", extent=extent,
                           cmap="viridis")
    ax_mrs.plot([x[pixel_cut], x[pixel_cut]], [y[0], y[-1]], "r--", lw=1.5)
    ax_mrs.set_title(r"Continiuum")
    plt.colorbar(im_mrs, ax=ax_mrs, fraction=0.046, pad=0.04)

    im_rec = ax_rec.imshow(img_rec_c, origin="lower", extent=extent,
                           cmap="viridis")
    ax_rec.plot([x[pixel_cut], x[pixel_cut]], [y[0], y[-1]], "r--", lw=1.5)
    ax_rec.set_title(r"Spectral Line")
    plt.colorbar(im_rec, ax=ax_rec, fraction=0.046, pad=0.04)

    for ax in [ax_mrs, ax_rec]:
        ax.set_xlabel(r"$\Delta$RA ($^{\prime\prime}$)")
        ax.set_ylabel(r"$\Delta$Dec ($^{\prime\prime}$)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    # ---- profils ----
    ax_prof2 = ax_prof.twinx()

    line1 = ax_prof.plot(
        y, prof_mrs,
        label="Continuum",
        lw=2,
        drawstyle="steps-mid",
        color="tab:blue"
    )

    line2 = ax_prof2.plot(
        y, prof_rec,
        label="Spectral Line",
        lw=2,
        drawstyle="steps-mid",
        color="tab:orange"
    )
    ax_prof.set_xlabel(r"$\Delta$Dec ($^{\prime\prime}$)", fontsize=12)
    ax_prof.set_ylabel("Continuum intensity", fontsize=12, color="tab:blue")
    ax_prof2.set_ylabel("Spectral line intensity", fontsize=12, color="tab:orange")

    ax_prof.tick_params(axis='y', labelcolor="tab:blue")
    ax_prof2.tick_params(axis='y', labelcolor="tab:orange")

    ax_prof.grid(True, alpha=0.4)
    ax_prof.legend(fontsize=11)
    ax_prof2.legend(fontsize=11)

    ax_prof.set_title(
        rf"Vertical cut at $x={x[pixel_cut]:.2f}''$ — "
        rf"$\lambda={wavel_axis_SS[idx]:.2f}\,\mu m$",
        fontsize=13,
    )
    plt.show()



wavelength_cuts = [7.7, 10.0, 11.3, 12.8, 15.0, 18.0, 21.0, 25.5]
filters = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
i = 0
for idx, wavelength_cut in enumerate(wavelength_cuts):

    pce_filt = pce_mirim_SS[idx]
    pce_norm = trapezoid(pce_filt, x=wavel_axis_SS)
    y_spectro_filt = trapezoid(raw_y_spectro*pce_filt[:,None,None], x=wavel_axis_SS, axis=0)/pce_norm
    mirim_filter = y_mirim[idx]


    idx = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))

    img_rec = res_cube[idx]
    img_mrs = raw_y_spectro[idx]

    # ---- crop top/bottom ----
    img_rec_c = img_rec[crop_tb:-crop_tb, crop_lr:-crop_lr]
    img_mrs_c = img_mrs[crop_tb:-crop_tb, crop_lr:-crop_lr]

    ny, nx = img_rec_c.shape

    # ---- axes en arcsec ----
    x = (np.arange(nx) - nx / 2) * pixscale
    y = (np.arange(ny) - ny / 2) * pixscale
    extent = [x[0], x[-1], y[0], y[-1]]

    # ---- profils ----
    prof_rec = img_rec_c[:, pixel_cut]
    prof_mrs = img_mrs_c[:, pixel_cut]

    # ---- layout ----
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax_mrs = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    # ---- images (même échelle) ----
    vmin = min(img_mrs_c.min(), img_rec_c.min())
    vmax = max(img_mrs_c.max(), img_rec_c.max())

    im_mrs = ax_mrs.imshow(img_mrs_c, origin="lower", extent=extent,
                           cmap="viridis", vmin=vmin, vmax=vmax)
    ax_mrs.plot([x[pixel_cut], x[pixel_cut]], [y[0], y[-1]], "r--", lw=1.5)
    ax_mrs.set_title(r"$\mathbf{y}_{\mathrm{MRS}}$")
    plt.colorbar(im_mrs, ax=ax_mrs, fraction=0.046, pad=0.04)

    im_rec = ax_rec.imshow(img_rec_c, origin="lower", extent=extent,
                           cmap="viridis", vmin=vmin, vmax=vmax)
    ax_rec.plot([x[pixel_cut], x[pixel_cut]], [y[0], y[-1]], "r--", lw=1.5)
    ax_rec.set_title(r"$\hat{\mathbf{x}}$ (Fused)")
    plt.colorbar(im_rec, ax=ax_rec, fraction=0.046, pad=0.04)

    for ax in [ax_mrs, ax_rec]:
        ax.set_xlabel(r"$\Delta$RA ($^{\prime\prime}$)")
        ax.set_ylabel(r"$\Delta$Dec ($^{\prime\prime}$)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    # ---- profils ----
    ax_prof.plot(y, prof_mrs, label="MRS", lw=2, drawstyle="steps-mid")
    ax_prof.plot(y, prof_rec, label="Fused", lw=2, drawstyle="steps-mid")
    ax_prof.set_xlabel(r"$\Delta$Dec ($^{\prime\prime}$)", fontsize=12)
    ax_prof.set_ylabel("Intensity", fontsize=12)
    ax_prof.grid(True, alpha=0.4)
    ax_prof.legend(fontsize=11)

    ax_prof.set_title(
        rf"Vertical cut at $x={x[pixel_cut]:.2f}''$ — "
        rf"$\lambda={wavel_axis_SS[idx]:.2f}\,\mu m$",
        fontsize=13,
    )
    # plt.savefig(
    #             f"/home/nmonnier/Presentations/20260123_INCLASS/Cut_{pixel_cut}_Fusion_MRS_MIRIM_{wavel_axis_SS[idx]:.2f}um.png",
    #             dpi=200, bbox_inches="tight"
    #         )
    plt.show()


plot_cube(res_cube, wavel_axis_SS, title='Fused Cube', show=False)
plot_cube(raw_y_spectro, wavel_axis_SS, title='Raw MRS Cube', show=False)


plt.show()
