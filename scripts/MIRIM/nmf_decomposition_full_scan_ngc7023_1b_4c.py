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
# wavel_axis2 = np.array(hdul[5].data[0])[0, :, 0]

raw_cube = np.load("/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/Channels/rotated_merged_cube.npy")
wavel_axis = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavelength/simplified_model_wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy')
mirim_ref_wavel_axis = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavelength/full_scan_wavel_axis.npy')
print(f"Raw cube shape: {raw_cube.shape}")
print(f"Wavelength axis shape: {wavel_axis.shape}")
# print(f"Wavelength axis2 shape: {wavel_axis2.shape}")
print(f"Mirim reference wavelength axis shape: {mirim_ref_wavel_axis.shape}")
print(f"Min/Max wavelength: {wavel_axis.min()}/{wavel_axis.max()}")
# print(f"Min/Max wavelength2: {wavel_axis2.min()}/{wavel_axis2.max()}")
print(f"Min/Max Mirim reference wavelength: {mirim_ref_wavel_axis.min()}/{mirim_ref_wavel_axis.max()}")

# raise NotImplementedError("Stop here to check the loaded cube and wavelength axis before proceeding with the analysis.")

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

""" Spectre moyen & détection des raies spectrales """
mean_spectrum = np.nanmean(foi_cube, axis=(1, 2))
plt.figure()
plt.plot(wavel_axis, mean_spectrum)
plt.show()

baseline = iterative_baseline_removal(
    mean_spectrum, lam=1e3, ncycles=5, sigma=3.0
)
baseline_sub = mean_spectrum - baseline
plt.figure()
plt.plot(mean_spectrum)
plt.title("Mean Spectrum")
plt.figure()
plt.plot(baseline_sub)
plt.title("Baseline Subtracted Spectrum")
plt.figure()
plt.plot(baseline)
plt.title("Baseline")
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

plt.figure()
plt.plot(wavel_axis, np.mean(spectral_line_cube, axis=(1, 2)))
plt.title("Mean Spectral Line Cube")


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

for sig in signals_sep:
    plt.figure()
    plt.plot(wavel_axis, sig)
    plt.title("Separated Spectral Line")
# plt.show()



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


"""
Sélection des longueurs d'ondes utiles pour la fusion MIRIM (SS de 4, sans les raies spectrales)
"""

wavel_axis = wavel_axis[::4]
cube_interp = cube_interp[::4, :, :]



""" Décomposition NMF """
data_matrix = rearrange(cube_interp, "L I J -> (I J) L")

nmf = NMF(
    n_components=5,
    init="random",
    random_state=0,
    max_iter=1000
)
nmf.fit(data_matrix)

components = nmf.components_

# raise NotImplementedError("Stop here to avoid running the NMF decomposition before checking the filtered cube and the separated spectral lines.")


""" Ajout des raies spectrales aux composantes"""
thin_components = np.zeros((components.shape[0], len(wavel_axis)))

for i in range(components.shape[0]):
    thin_components[i] = components[i]

# Comment bellow if you don't want to add spectrale lines to NMF components (e.g. for MIRIM fusion)
# for sig in signals_sep:
#     thin_components = np.vstack([thin_components, sig])


# plt.figure()
# for i in range(thin_components.shape[0]):
#     plt.plot(wavel_axis, thin_components[i], label=f'Component {i+1}')
# plt.title("NMF Components with Spectral Lines")
# plt.legend()
# plt.show()

""" Sauvegarde des résultats """
template_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/'
np.save(template_dir + f'NMF/full_scan_simplified_{mrs_lim_band[0]}_to_{mrs_lim_band[1]}_{components.shape[0]}_nmf_components_mirim_fusion.npy', thin_components)
# np.save(template_dir + f'wavelength/{mrs_lim_band[0]}_to_{mrs_lim_band[1]}_wavel_axis_SS4_mirim_fusion.npy', wavel_axis)
