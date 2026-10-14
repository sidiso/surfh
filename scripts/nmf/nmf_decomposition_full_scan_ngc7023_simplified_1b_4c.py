# %%
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import rotate, gaussian_filter, generic_filter
from scipy.signal import medfilt

from sklearn.decomposition import NMF
from einops import rearrange

from surfh.Signalprocessing import fitting, utilities, utils
from surfh.Signalprocessing.baseline import iterative_baseline_removal
from surfh.Models import wavelength_mrs
from surfh.Vizualisation.cube_vizualisation import plot_cube

""" Chargement des données fits."""
# fits_file = Path(
#     "/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/"
#     "Full_scan_NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits"
# )

# hdul = fits.open(fits_file)
# raw_cube = hdul[1].data
# wavel_axis = np.array(hdul[5].data[0])[0, :, 0]

raw_cube = np.load("/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/Channels/rotated_merged_cube.npy")
wavel_axis = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavelength/simplified_model_wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy')
print(f"Wavelength axis shape: {wavel_axis.shape}")
print(f"Min/Max wavelength: {wavel_axis.min()}/{wavel_axis.max()}")


""" Séléction des bandes MRS """
mrs_lim_band = ['ch1b', 'ch4c']
min_wavel = wavelength_mrs.get_mrs_wavelength(mrs_lim_band[0])[0]
max_wavel = wavelength_mrs.get_mrs_wavelength(mrs_lim_band[1])[-1]
start_idx = np.argmin(np.abs(wavel_axis - min_wavel))
end_idx = np.argmin(np.abs(wavel_axis - max_wavel))
print("Start index:", start_idx, "End index:", end_idx)

""" Rotation du cube"""
# rotation_angle = -37  # degrees

# raw_cube = np.nan_to_num(raw_cube, nan=0.0)
# rotated_cube = np.empty_like(raw_cube)

# for k in range(raw_cube.shape[0]):
#     rotated_cube[k] = rotate(
#         raw_cube[k],
#         angle=rotation_angle,
#         reshape=False,
#         axes=(1, 0),
#         order=3,
#         mode="nearest"
#     )

# """ Sélection du FoV commun aux 4 canaux """
# foi_cube = rotated_cube[:, 12:175, 68:95]

fusion_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'
wavel_axis2 = np.load(fusion_dir + 'Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy')
print(f"Wavelength axis shape after loading: {wavel_axis2.shape}")
print(f"Min/Max wavelength after loading: {wavel_axis2.min()}/{wavel_axis2.max()}")
mask = ~np.any(np.isclose(wavel_axis[:, None], wavel_axis2[None, :], rtol=1e-6, atol=1e-9), axis=1)
diff_a = wavel_axis[mask]
mask2 = ~np.any(np.isclose(wavel_axis2[:, None], wavel_axis[None, :], rtol=1e-6, atol=1e-9), axis=1)
diff_b = wavel_axis2[mask2]
diff_all = np.concatenate([diff_a, diff_b])
print(f"Difference are : {diff_all}")

spectral_cut = 200
wavel_axis2 = wavel_axis[start_idx:end_idx-spectral_cut]
wavel_axis = wavel_axis[start_idx:end_idx-spectral_cut]
print(wavel_axis)
thin_wavel = wavel_axis.copy()
# foi_cube = foi_cube[start_idx:end_idx, :, :]
foi_cube = raw_cube[:,19:231, 287:323].copy()
# np.save("/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/Channels/rotated_merged_foi_cube.npy", foi_cube)



foi_cube = foi_cube[start_idx:end_idx-spectral_cut, :, :]

plot_cube(foi_cube, wavel_axis, title="FoV Cube after Rotation and Band Selection", show=True)

# %%

# %%

# %%

# %%
""" Spectre moyen & détection des raies spectrales """
mean_spectrum = np.nanmean(foi_cube, axis=(1, 2))
plt.figure()
plt.plot(wavel_axis, mean_spectrum)
plt.show()

baseline = iterative_baseline_removal(
    mean_spectrum, lam=1e3, ncycles=5, sigma=3.0
)
baseline_sub = mean_spectrum - baseline
# plt.figure()
# plt.plot(mean_spectrum)
# plt.title("Mean Spectrum")
# plt.figure()
# plt.plot(baseline_sub)
# plt.title("Baseline Subtracted Spectrum")
# plt.figure()
# plt.plot(baseline)
# plt.title("Baseline")
# plt.show()
mad = utilities.sliding_mad(baseline_sub)

fitted_peaks = fitting.detect_and_fit_peaks(
    baseline_sub, mad, sigma=5, distance=5, wavelength=wavel_axis
)

clean_peaks, mean_sigma, std_sigma = fitting.filter_clean_peaks(fitted_peaks)
peak_indices = [p["peak_index"] for p in clean_peaks]

print(f"Clean peaks: {len(clean_peaks)}")
print(f"Mean sigma: {mean_sigma:.3f}, Std sigma: {std_sigma:.3f}")



""" Insertion des longueurs d’onde interpolées au niveau des centres des raies détectées """
for peak in sorted(clean_peaks, key=lambda p: p["center"], reverse=True):
    center = peak["center"]
    i0, i1 = int(np.floor(center)), int(np.ceil(center))
    frac = center - i0

    w_center = wavel_axis[i0] * (1 - frac) + wavel_axis[i1] * frac
    thin_wavel = np.insert(thin_wavel, i1, w_center)


""" Fit pixel par pixel"""
L, I, J = foi_cube.shape
continuum_cube = np.zeros_like(foi_cube)
spectral_line_cube = np.zeros_like(foi_cube)

CHUNK_SIZE = 46
args = [
    (i, j, foi_cube, peak_indices, mean_sigma, std_sigma)
    for i in range(I)
    for j in range(J)
]

chunks = [args[k:k + CHUNK_SIZE] for k in range(0, len(args), CHUNK_SIZE)]

start = time.time()
with ProcessPoolExecutor() as executor:
    for chunk_result in executor.map(utils.process_pixel_chunk, chunks):
        for i, j, cont, line in chunk_result:
            continuum_cube[:, i, j] = cont
            spectral_line_cube[:, i, j] = line

print(f"Fitting time: {time.time() - start:.2f}s")

# plt.figure()
# plt.plot(wavel_axis, np.mean(spectral_line_cube, axis=(1, 2)))
# plt.title("Mean Spectral Line Cube")


import numpy as np
from scipy.signal import find_peaks

# signal : ton signal 1D
# x : axe (optionnel)

# 1. détecter les pics
peaks, _ = find_peaks(np.mean(spectral_line_cube, axis=(1, 2)), height=np.max(np.mean(spectral_line_cube, axis=(1, 2)))*0.03, distance=50)

# 2. largeur approximative d'une gaussienne (à ajuster)
half_width = 40

signals_sep = []

for p in peaks:
    s = np.zeros_like(np.mean(spectral_line_cube, axis=(1, 2)))
    start = max(0, p - half_width)
    end   = min(len(np.mean(spectral_line_cube, axis=(1, 2))), p + half_width)
    s[start:end] = np.mean(spectral_line_cube, axis=(1, 2))[start:end]
    signals_sep.append(s)



_, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# Top panel: continuum vs. baseline
axs[0].plot(wavel_axis, np.nanmean(continuum_cube, axis=(1, 2)) + np.nanmean(spectral_line_cube, axis=(1, 2)), label='Separated Continuum', color='red', alpha = 0.75)
axs[0].plot(wavel_axis, np.nanmean(continuum_cube, axis=(1, 2)), label='Input Spectrum', color='black', alpha=0.75)
axs[0].legend()

# --- Bottom panel: Gaussian fits over extracted lines ---
axs[1].plot(wavel_axis, np.nanmean(foi_cube, axis=(1, 2)) - baseline, color='black', label='Separated lines', alpha=0.75)
axs[1].plot(wavel_axis, np.nanmean(spectral_line_cube, axis=(1, 2)), color='teal', label='Gaussians (best fit)', alpha=0.75)
axs[1].plot(wavel_axis, np.nanmean(foi_cube, axis=(1, 2)) - baseline - np.nanmean(spectral_line_cube, axis=(1, 2)) - 5e2, color='crimson', label='Residuals (shifted)', alpha=0.75)
axs[1].set_xlabel("Wavelength (microns)")
axs[1].legend()

plt.tight_layout()
plt.show()


""" Nettoyage du continuum """
# continuum_cube[continuum_cube < 0] = np.nan
print(f"continuum_cube dtype is {continuum_cube.dtype}")
filtered_cube = np.empty_like(continuum_cube)
for i in range(I):
    for j in range(J):
        spectrum = continuum_cube[:, i, j]
        temp = np.nan_to_num(spectrum, nan=np.nanmedian(spectrum)).astype(np.float32)
        filtered_cube[:, i, j] = medfilt(temp, kernel_size=5)

filtered_cube[filtered_cube < 0] = np.nan






#TODEL: Build cube filtered with Spectral lines
import pathlib
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits
filter_cube_tosave = filtered_cube.copy()
for p in peaks:
    s = np.zeros_like(np.mean(spectral_line_cube, axis=(1, 2)))
    start = max(0, p - half_width)
    end   = min(len(np.mean(spectral_line_cube, axis=(1, 2))), p + half_width)
    s[start:end] = np.mean(spectral_line_cube, axis=(1, 2))[start:end]
    filter_cube_tosave[start:end, :, :] = foi_cube[start:end, :, :]
# path = pathlib.Path('/home/nmonnier/Data/JWST/NGC_7023/Fusion/')
# path.mkdir(parents=True, exist_ok=True)
# metadata = {"WAVELENGTH": wavel_axis}
# save_numpy_to_fits(filter_cube_tosave, metadata, path/'Filtered_raw_cube.fits', masks=None)







""" Interpolation & lissage 3D"""
mask = np.isfinite(filtered_cube)
cube_filled = np.nan_to_num(filtered_cube, nan=0.0)

sigma = (1, 2, 2)
smoothed = gaussian_filter(cube_filled, sigma=sigma, mode="nearest")
weights = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")

cube_interp = smoothed / weights
cube_interp[~np.isfinite(cube_interp)] = 0.0
cube_interp[cube_interp < 0] = 0.0


# %%

# %%

# %%

# %%

# %%


""" Décomposition NMF """
data_matrix = rearrange(cube_interp, "L I J -> (I J) L")

# nmf = NMF(
#     n_components=5,
#     init="random",
#     random_state=0,
#     max_iter=1000
# )
# nmf.fit(data_matrix)

# components = nmf.components_
# W = nmf.fit_transform(data_matrix)
# H = nmf.components_
# nmf_recons = W @ H
# # # %%

# # %%

# plt.figure()
# plt.plot(wavel_axis, np.mean(data_matrix, axis=(0)), label="Original")
# plt.plot(wavel_axis, np.mean(nmf_recons, axis=(0)), label="NMF Reconstruction")
# plt.legend()
# plt.title("NMF Reconstruction vs Original")


# plt.figure()
# plt.plot(wavel_axis, np.mean((data_matrix-nmf_recons)/data_matrix, axis=(0)))
# plt.title("Relative Error")
# plt.show()  



n_components_list = range(1, 30)
errors = []
errors2 = []
relative_errors = []

for n in n_components_list:
    nmf = NMF(
        n_components=n,
        init="random",
        random_state=None,
        max_iter=1000
    )
    W = nmf.fit_transform(data_matrix)
    H = nmf.components_
    
    # Méthode 1 : erreur sklearn
    err = nmf.reconstruction_err_
    
    # Méthode 2 (équivalente) :
    err2 = np.mean(data_matrix, axis=0) - np.mean(W @ H, axis=0)
    rel_err = err2/np.mean(W @ H, axis=0)

    errors.append(err)
    errors2.append(err2)
    relative_errors.append(rel_err)



# %%

# %%

# %%

#%% 
# # Plot
# plt.figure()
# plt.plot(n_components_list, errors, marker='o')
# plt.xlabel("n_components")
# plt.ylabel("Reconstruction error")
# plt.title("NMF error vs number of components")
# plt.grid()

# plt.figure()
# plt.plot(n_components_list, relative_errors, marker='o')
# plt.xlabel("n_components")
# plt.ylabel("Reconstruction relative error (%)")
# plt.title("NMF relative error vs number of components (manual)")
# plt.grid()

# Style global
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# Taille adaptée A&A (simple column ~ 8.8 cm)
fig, ax = plt.subplots(figsize=(3.5, 2.8))  # pouces

ax.plot(
    n_components_list,
    np.abs(np.mean(np.array(relative_errors), axis=1)*100),
    marker='o',
    markersize=3,
    linewidth=1,
    color='black',
    label="NMF reconstruction"
)

ax.set_xlabel("Number of Spectral templates")
ax.set_ylabel("Relative reconstruction error (%)")

# Optionnel mais souvent utile
# ax.set_xlim(min(n_components_list), max(n_components_list))
# ax.set_ylim(-max(abs(relative_errors))*1.05, max(abs(relative_errors))*1.05)
ax.set_yscale('log')

# Grid discret (A&A préfère léger ou absent)
ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

# ticks vers l’intérieur (style astro)
ax.tick_params(direction='in', top=True, right=True)

# marges serrées
plt.tight_layout()

# plt.savefig(
#             f"/home/nmonnier/Presentations/Fusion_figures/NMF_Reconstruction_error.png",
#             dpi=300, bbox_inches="tight"
#         )
# plt.savefig("nmf_error_vs_components.pdf", bbox_inches='tight')
plt.show()
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

#%%

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)

# ---- Spectres ----
axes[0].plot(wavel_axis, np.mean(W @ H, axis=0), label=r"NMF", lw=2, linestyle="dotted")
axes[0].plot(wavel_axis, np.mean(data_matrix, axis=0), label=r"MRS", lw=2, linestyle="--")
axes[0].set_ylabel("Flux (MJy/sr)", fontsize=14)
# axes[0].set_title("Mean MRS Spectrum", fontsize=14)
axes[0].legend(fontsize=13)
axes[0].grid(True, alpha=0.4)

# ---- Erreur relative ----
axes[1].plot(wavel_axis, errors2, 
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



# raise NotImplementedError("Stop here to avoid running the NMF decomposition before checking the filtered cube and the separated spectral lines.")
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

#%%

""" Ajout des raies spectrales aux composantes"""
thin_components = np.zeros((components.shape[0], len(wavel_axis)))

for i in range(components.shape[0]):
    thin_components[i] = components[i]

for sig in signals_sep:
    thin_components = np.vstack([thin_components, sig])


plt.figure()
for i in range(thin_components.shape[0]):
    plt.plot(wavel_axis, thin_components[i], label=f'Component {i+1}')
plt.title("NMF Components with Spectral Lines")
plt.legend()
plt.show()

""" Sauvegarde des résultats """
template_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/'
# np.save(template_dir + f'NMF/full_scan_simplified_{mrs_lim_band[0]}_to_{mrs_lim_band[1]}_{components.shape[0]}_nmf_components_and_{thin_components.shape[0]-components.shape[0]}_spectral_lines.npy', thin_components)
# np.save(template_dir + f'wavelength/{mrs_lim_band[0]}_to_{mrs_lim_band[1]}_wavel_axis_with_spectral_line.npy', thin_wavel)

# %%
