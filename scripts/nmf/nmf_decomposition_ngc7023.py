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

from rich.progress import Progress

from surfh.Signalprocessing import fitting, utilities, utils
from surfh.Signalprocessing.baseline import iterative_baseline_removal
from surfh.Models import wavelength_mrs

""" Chargement des données fits."""
fits_file = Path(
    "/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/"
    "ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits"
)

hdul = fits.open(fits_file)
raw_cube = hdul[1].data
wavel_axis = np.array(hdul[5].data[0])[0, :, 0]


""" Séléction des bandes MRS """
mrs_lim_band = ['ch1a', 'ch4b']
min_wavel = wavelength_mrs.get_mrs_wavelength(mrs_lim_band[0])[0]
max_wavel = wavelength_mrs.get_mrs_wavelength(mrs_lim_band[1])[-1]
start_idx = np.argmin(np.abs(wavel_axis - min_wavel))
end_idx = np.argmin(np.abs(wavel_axis - max_wavel))
print("Start index:", start_idx, "End index:", end_idx)

""" Rotation du cube"""
rotation_angle = -37  # degrees

raw_cube = np.nan_to_num(raw_cube, nan=0.0)
rotated_cube = np.empty_like(raw_cube)

for k in range(raw_cube.shape[0]):
    rotated_cube[k] = rotate(
        raw_cube[k],
        angle=rotation_angle,
        reshape=False,
        axes=(1, 0),
        order=3,
        mode="nearest"
    )

""" Sélection du FoV commun aux 4 canaux """
foi_cube = rotated_cube[:, 20:82, 36:64]

wavel_axis = wavel_axis[start_idx:end_idx]
thin_wavel = wavel_axis.copy()
foi_cube = foi_cube[start_idx:end_idx, :, :]


""" Spectre moyen & détection des raies spectrales """
mean_spectrum = np.nanmean(foi_cube, axis=(1, 2))

baseline = iterative_baseline_removal(
    mean_spectrum, lam=1e3, ncycles=5, sigma=3.0
)
baseline_sub = mean_spectrum - baseline
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

""" Interpolation & lissage 3D"""
mask = np.isfinite(filtered_cube)
cube_filled = np.nan_to_num(filtered_cube, nan=0.0)

sigma = (1, 2, 2)
smoothed = gaussian_filter(cube_filled, sigma=sigma, mode="nearest")
weights = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")

cube_interp = smoothed / weights
cube_interp[~np.isfinite(cube_interp)] = 0.0
cube_interp[cube_interp < 0] = 0.0

""" Décomposition NMF """
data_matrix = rearrange(cube_interp, "L I J -> (I J) L")

nmf = NMF(
    n_components=6,
    init="random",
    random_state=0,
    max_iter=1000
)
nmf.fit(data_matrix)

components = nmf.components_

""" Ajout des raies spectrales aux composantes"""
thin_components = np.zeros((components.shape[0], len(thin_wavel)))

for i in range(components.shape[0]):
    thin_components[i] = np.interp(thin_wavel, wavel_axis, components[i])

for k, peak in enumerate(clean_peaks):
    peak_line = np.zeros(len(thin_wavel))
    idx = np.argmin(np.abs(thin_wavel - peak["wavel_center"]))
    peak_line[idx] = mean_spectrum[peak_indices[k]]
    thin_components = np.vstack([thin_components, peak_line])


""" Sauvegarde des résultats """
template_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/'
np.save(template_dir + f'NMF/{mrs_lim_band[0]}_to_{mrs_lim_band[1]}_{components.shape[0]}_nmf_components_and_{thin_components.shape[0]-components.shape[0]}_spectral_lines.npy', thin_components)
np.save(template_dir + f'wavelength/{mrs_lim_band[0]}_to_{mrs_lim_band[1]}_wavel_axis_with_spectral_line.npy', thin_wavel)
