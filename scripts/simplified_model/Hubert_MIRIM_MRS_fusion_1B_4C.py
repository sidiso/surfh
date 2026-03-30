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
from scipy.integrate import trapezoid

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

L_SS = 5 # Sous-échantillonnage spectrale pour la simulation

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
raw_y_spectro = np.load('/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/Channels/rotated_merged_foi_cube.npy')
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
plt.figure()
for i, filt in enumerate(['F770W', 'F1000W', 'F1130W', 'F1280W','F1500W', 'F1800W', 'F2100W', 'F2550W']):
    print(pce_mirim_SS.shape)
    plt.plot(wavel_axis_SS, pce_mirim_SS[i])
plt.show()

corrected_y_mirim = y_mirim.copy()

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


# raise SystemExit
""""""""""""""""""""""""""""""""""""""""""""""""""
start = time.time()
quadcriterion = QuadCriterion2(
    mu_imager,
    y_mirim,
    mirim_model_for_fusion,
    list_mu_spectro[15],
    y_spectro,
    spectro_model,
    list_mu_reg[2],
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

y_mrs_mean_spectrum = np.mean(raw_y_spectro[:,2:-2,2:-2], axis=(1,2))
xy_mirim = mirim_model_for_fusion.forward(quadcrit_rec_maps)


# Flip all data
for i in range(xy_mirim.shape[0]):
    xy_mirim[i] = np.fliplr(xy_mirim[i])
    y_mirim[i] = np.fliplr(y_mirim[i])
for i in range(res_cube.shape[0]):
    res_cube[i] = np.fliplr(res_cube[i])
    raw_y_spectro[i] = np.fliplr(raw_y_spectro[i])





mean_rec = np.mean(res_cube[:,2:-2,2:-2], axis=(1, 2))
rel_err = (mean_rec - y_mrs_mean_spectrum) / y_mrs_mean_spectrum

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)

# ---- Spectres ----
axes[0].plot(wavel_axis_SS, mean_rec, label="Reconstructed Spectrum", lw=2, linestyle="dotted")
axes[0].plot(wavel_axis_SS, y_mrs_mean_spectrum, label="Raw MRS Spectrum", lw=2, linestyle="--")
axes[0].set_ylabel("Mean Flux", fontsize=12)
axes[0].set_title("Mean MRS Spectrum", fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.4)

# ---- Erreur relative ----
axes[1].plot(wavel_axis_SS, rel_err, label="Relative Error", lw=2)
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
axes[1].set_ylabel("Relative Error", fontsize=12)
axes[1].set_title("Relative Spectral Error", fontsize=14)
axes[1].grid(True, alpha=0.4)

# ---- ticks plus gros ----
for ax in axes:
    ax.tick_params(axis="both", labelsize=11)

print("Mean absolute relative error spectrum:",
      np.mean(np.abs(rel_err)))
# plt.savefig(
#             f"/home/nmonnier/Presentations/20260123_INCLASS/Fusion_MRS_MIRIM_{name}_MeanFlux.png",
#             dpi=200, bbox_inches="tight"
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


# --- images cropées ---
img0 = crop2(raw_y_spectro[idx_filter], crop)
img1 = crop2(res_cube[idx_filter], crop)
img2 = crop2(y_mirim[0], crop)
img3 = crop2(xy_mirim[0], crop)
img4 = crop2(y_mirim[0] - xy_mirim[0], crop)
img5 = crop2(y_mirim[0] - y_spectro_filt, crop)

imgs = [img0, img1, img2, img3, img4, img5]
titles = [
    fr"$\mathbf{{y}}_{{\mathrm{{MRS}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",
    fr"$\hat{{\mathbf{{x}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",    
    fr"$\mathbf{{y}}_{{\mathrm{{MIRIM}}}}$ — Filter 770W",
    fr"$\hat{{\mathbf{{y}}}}_{{\mathrm{{MIRIM}}}}$ — Filter 770W",
    "Difference ($\mathbf{{y}}_{{\mathrm{{MIRIM}}}} - \hat{{\mathbf{{y}}}}_{{\mathrm{{MIRIM}}}}$)",
    "Y-Ymrs"
]

ny, nx = img0.shape

# --- axes en arcsec centrés ---
x = (np.arange(nx) - nx / 2) * pixscale
y = (np.arange(ny) - ny / 2) * pixscale
extent = [x[0], x[-1], y[0], y[-1]]

# --- figure ---
fig, axes = plt.subplots(1, 6, figsize=(18, 5), constrained_layout=True)

for ax, img, title in zip(axes, imgs, titles):

    im = ax.imshow(img, origin="lower", extent=extent, cmap="viridis")
    ax.set_title(title, fontsize=10)

    ax.set_xlabel("ΔRA (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_aspect("equal")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.savefig(
#             f"/home/nmonnier/Presentations/20260123_INCLASS/Fusion_MRS_MIRIM_{name}_F0770W.png",
#             dpi=200, bbox_inches="tight"
#         )
# plt.show()


filter_wavelength = 15.0
idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
pce_filt = pce_mirim_SS[4]
pce_norm = trapezoid(pce_filt, x=wavel_axis_SS)
y_spectro_filt = trapezoid(raw_y_spectro*pce_filt[:,None,None], x=wavel_axis_SS, axis=0)/pce_norm

filters_name = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']

# --- images cropées ---
img0 = crop2(raw_y_spectro[idx_filter], crop)
img1 = crop2(res_cube[idx_filter], crop)
img2 = crop2(y_mirim[4], crop)
img3 = crop2(xy_mirim[4], crop)
img4 = crop2(y_mirim[4] - xy_mirim[4], crop)
img5 = crop2(y_mirim[4] - y_spectro_filt, crop)


imgs = [img0, img1, img2, img3, img4, img5]
titles = [
    fr"$\mathbf{{y}}_{{\mathrm{{MRS}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",
    fr"$\hat{{\mathbf{{x}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",    
    fr"$\mathbf{{y}}_{{\mathrm{{MIRIM}}}}$ — Filter 1500W",
    fr"$\hat{{\mathbf{{y}}}}_{{\mathrm{{MIRIM}}}}$ — Filter 1500W",
    "Difference ($\mathbf{{y}}_{{\mathrm{{MIRIM}}}} - \hat{{\mathbf{{y}}}}_{{\mathrm{{MIRIM}}}}$)",
    "Y-Ymrs"

]

ny, nx = img0.shape

# --- axes en arcsec centrés ---
x = (np.arange(nx) - nx / 2) * pixscale
y = (np.arange(ny) - ny / 2) * pixscale
extent = [x[0], x[-1], y[0], y[-1]]

# --- figure ---
fig, axes = plt.subplots(1, 6, figsize=(18, 5), constrained_layout=True)

for ax, img, title in zip(axes, imgs, titles):

    im = ax.imshow(img, origin="lower", extent=extent, cmap="viridis")
    ax.set_title(title, fontsize=10)

    ax.set_xlabel("ΔRA (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_aspect("equal")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.savefig(
#             f"/home/nmonnier/Presentations/20260123_INCLASS/Fusion_MRS_MIRIM_{name}_F1500W.png",
#             dpi=200, bbox_inches="tight"
#         )
# plt.show()


filter_wavelength = 25.5
idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
pce_filt = pce_mirim_SS[-1]
pce_norm = trapezoid(pce_filt, x=wavel_axis_SS)
y_spectro_filt = trapezoid(raw_y_spectro*pce_filt[:,None,None], x=wavel_axis_SS, axis=0)/pce_norm
filters_name = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']

# --- images cropées ---
img0 = crop2(raw_y_spectro[idx_filter], crop)
img1 = crop2(res_cube[idx_filter], crop)
img2 = crop2(y_mirim[-1], crop)
img3 = crop2(xy_mirim[-1], crop)
img4 = crop2(y_mirim[-1] - xy_mirim[-1], crop)
img5 = crop2(y_mirim[-1] - y_spectro_filt, crop)

imgs = [img0, img1, img2, img3, img4, img5]
titles = [
    fr"$\mathbf{{y}}_{{\mathrm{{MRS}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",
    fr"$\hat{{\mathbf{{x}}}}$ — $\lambda = {wavel_axis_SS[idx_filter]:.2f}\,\mu m$",    
    fr"$\mathbf{{y}}_{{\mathrm{{MIRIM}}}}$ — Filter 2550W",
    fr"$\hat{{\mathbf{{y}}}}_{{\mathrm{{MIRIM}}}}$ — Filter 2550W",
    "Difference ($\mathbf{{y}}_{{\mathrm{{MIRIM}}}} - \hat{{\mathbf{{y}}}}_{{\mathrm{{MIRIM}}}}$)",
    "Y-Ymrs"
]

ny, nx = img0.shape

# --- axes en arcsec centrés ---
x = (np.arange(nx) - nx / 2) * pixscale
y = (np.arange(ny) - ny / 2) * pixscale
extent = [x[0], x[-1], y[0], y[-1]]

# --- figure ---
fig, axes = plt.subplots(1, 6, figsize=(18, 5), constrained_layout=True)

for ax, img, title in zip(axes, imgs, titles):

    im = ax.imshow(img, origin="lower", extent=extent, cmap="viridis")
    ax.set_title(title, fontsize=10)

    ax.set_xlabel("ΔRA (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_aspect("equal")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.savefig(
#             f"/home/nmonnier/Presentations/20260123_INCLASS/Fusion_MRS_MIRIM_{name}_F2550W.png",
#             dpi=200, bbox_inches="tight"
#         )
plt.show()


# filter_wavelength = 7.7
# idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
# fig, axes = plt.subplots(3, 3, figsize=(24, 8))
# plt.subplots_adjust(wspace=0.15, hspace=0.25)
# im0 = axes[0, 0].imshow(raw_y_spectro[idx_filter], origin='lower', cmap='viridis')
# axes[0, 0].set_title(f'MRS Raw Data - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im0, ax=axes[0, 0], fraction=0.025, pad=0.02)

# im1 = axes[0, 1].imshow(res_cube[idx_filter], origin='lower', cmap='viridis')
# axes[0, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im1, ax=axes[0, 1], fraction=0.025, pad=0.02)

# im2 = axes[0, 2].imshow(y_mirim[1], origin='lower', cmap='viridis')
# axes[0, 2].set_title(f'MIRIM - Filter 770W')
# plt.colorbar(im2, ax=axes[0, 2], fraction=0.025, pad=0.02)

# filter_wavelength = 12.8
# idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
# im3 = axes[1, 0].imshow(raw_y_spectro[idx_filter], origin='lower', cmap='viridis')
# axes[1, 0].set_title(f'MRS Raw Data - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im3, ax=axes[1, 0], fraction=0.025, pad=0.02)

# im4 = axes[1, 1].imshow(res_cube[idx_filter], origin='lower', cmap='viridis')
# axes[1, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im4, ax=axes[1, 1], fraction=0.025, pad=0.02)
# im5 = axes[1, 2].imshow(y_mirim[4], origin='lower', cmap='viridis')
# axes[1, 2].set_title(f'MIRIM - Filter F1280W')
# plt.colorbar(im5, ax=axes[1, 2], fraction=0.025, pad=0.02)


# filter_wavelength = 21.0
# idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
# im3 = axes[2, 0].imshow(raw_y_spectro[idx_filter], origin='lower', cmap='viridis')
# axes[2, 0].set_title(f'MRS Raw Data - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im3, ax=axes[2, 0], fraction=0.025, pad=0.02)

# im4 = axes[2, 1].imshow(res_cube[idx_filter], origin='lower', cmap='viridis')
# axes[2, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im4, ax=axes[2, 1], fraction=0.025, pad=0.02)
# im5 = axes[2, 2].imshow(y_mirim[-1], origin='lower', cmap='viridis')
# axes[2, 2].set_title(f'MIRIM - Filter F21000W')
# plt.colorbar(im5, ax=axes[2, 2], fraction=0.025, pad=0.02)
# # plt.tight_layout()


for filter in range(y_mirim.shape[0]):
    print(f'Filter {filters_name[filter]} - Flux ratio (xy_mirim/y_mirim): {np.sum(xy_mirim[filter][50:150, 3:-3])/np.sum(y_mirim[filter][50:150, 3:-3])}')


filters_name = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
fig, axes = plt.subplots(3, 8, figsize=(15, 5))
i = 0
for filter in range(y_mirim.shape[0]):


    im = axes[0, filter].imshow(y_mirim[filter][2:-2, 2:-2], origin='lower', cmap='viridis')
    axes[0, filter].set_title(f'Filter {filters_name[filter]}')
    plt.colorbar(im, ax=axes[0, filter], fraction=0.046, pad=0.04)

    im2 = axes[1, filter].imshow(xy_mirim[filter][2:-2, 2:-2], origin='lower', cmap='viridis')
    axes[1, filter].set_title(f'Reconstructed Filter {filters_name[filter]}')
    plt.colorbar(im2, ax=axes[1, filter], fraction=0.046, pad=0.04)
    im3 = axes[2, filter].imshow((y_mirim[filter][2:-2, 2:-2]- xy_mirim[filter][2:-2, 2:-2])/y_mirim[filter][2:-2, 2:-2], origin='lower', cmap='RdBu_r')
    axes[2, filter].set_title(f'Residuals Filter {filters_name[filter]}')
    plt.colorbar(im3, ax=axes[2, filter], fraction=0.046, pad=0.04)

    print(f'Filter {filters_name[filter]} - Mean absolute relative residual: {np.mean(np.abs((y_mirim[filter][2:-2, 2:-2]- xy_mirim[filter][2:-2, 2:-2])/y_mirim[filter][2:-2, 2:-2]))}')
    i = i+1
# plt.tight_layout()



pixscale = 0.1
crop_tb = 75
crop_lr = 3
pixel_cut = 12-4

wavelength_cuts = [7.7, 10.0, 11.3, 12.8, 15.0, 18.0, 21.0, 25.5]
filters = ['F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']

i = 0
for wavelength_cut in wavelength_cuts:



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
    ax_rec.set_title(r"$\hat{\mathbf{x}}$ (Reconstructed)")
    plt.colorbar(im_rec, ax=ax_rec, fraction=0.046, pad=0.04)

    for ax in [ax_mrs, ax_rec]:
        ax.set_xlabel(r"$\Delta$RA ($^{\prime\prime}$)")
        ax.set_ylabel(r"$\Delta$Dec ($^{\prime\prime}$)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    # ---- profils ----
    ax_prof.plot(y, prof_mrs, label="MRS", lw=2, drawstyle="steps-mid")
    ax_prof.plot(y, prof_rec, label="Reconstructed", lw=2, drawstyle="steps-mid")
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
# # Coupe vericale du cube à une lingueur d'onde donnée  et comparaison avec les données MRS brutes et le filtre MIRIM correspondant
# pixel_cut = 12
# wavelength_cut = 7.7
# idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
# cube_cut = res_cube[idx_wavelength_cut, :, :]
# plt.figure()
# plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 7.7 um', drawstyle='steps-mid')
# plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 7.7 um', drawstyle='steps-mid')
# plt.plot(y_mirim[1][:, pixel_cut], label='MIRIM Data Filter F770W Cut at 7.7 um', drawstyle='steps-mid')
# plt.xlabel('Pixel Y')
# plt.ylabel('Intensity')
# plt.title('Vertical Cut at X=18')
# plt.legend()    

# # Coupe à 21.0 um
# wavelength_cut = 21.0
# idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
# cube_cut = res_cube[idx_wavelength_cut, :, :]
# plt.figure()
# plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 21.0 um', drawstyle='steps-mid')
# plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 21.0 um', drawstyle='steps-mid')
# plt.plot(y_mirim[-1][:, pixel_cut], label='MIRIM Data Filter F2100W Cut at 21.0 um', drawstyle='steps-mid')
# plt.xlabel('Pixel Y')
# plt.ylabel('Intensity')
# plt.title('Vertical Cut at X=18')
# plt.legend()


# # Coupe à 18.0 um
# wavelength_cut = 18.0
# idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
# cube_cut = res_cube[idx_wavelength_cut, :, :]
# plt.figure()
# plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 18.0 um', drawstyle='steps-mid')
# plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 18.0 um', drawstyle='steps-mid')
# plt.plot(y_mirim[-1][:, pixel_cut], label='MIRIM Data Filter F2100W Cut at 18.0 um', drawstyle='steps-mid')
# plt.xlabel('Pixel Y')
# plt.ylabel('Intensity')
# plt.title('Vertical Cut at X=18')
# plt.legend()



# # Coupe à 15.0 um
# wavelength_cut = 15.0
# idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
# cube_cut = res_cube[idx_wavelength_cut, :, :]
# plt.figure()
# plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 15.0 um', drawstyle='steps-mid')
# plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 15.0 um', drawstyle='steps-mid')
# # plt.plot(y_mirim[-1][:, pixel_cut], label='MIRIM Data Filter F2100W Cut at 15.0 um', drawstyle='steps-mid')
# plt.xlabel('Pixel Y')
# plt.ylabel('Intensity')
# plt.title('Vertical Cut at X=18')
# plt.legend()



# # Show cut in the images
# plt.figure()
# plt.imshow(cube_cut, origin='lower', cmap='viridis')
# plt.axvline(x=pixel_cut, color='r', linestyle='--', label='Cut Position X=18')
# plt.title(f'Reconstructed Cube at {wavel_axis_SS[idx_wavelength_cut]:.2f} um')
# plt.colorbar(label='Intensity') 

plot_cube(res_cube, wavel_axis_SS, title='Reconstructed Cube', show=False)
plot_cube(raw_y_spectro, wavel_axis_SS, title='Raw MRS Cube', show=False)


plt.show()
