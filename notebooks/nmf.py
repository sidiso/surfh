import numpy as np
from astropy.io import fits
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import ndimage

from sklearn.decomposition import NMF
from einops import rearrange
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

from utils import iterative_baseline_removal, detect_and_fit_peaks, sliding_mad, filter_clean_peaks, build_line_spectrum, build_line_spectrum_from_gaussians, fit_peaks_only
from rich.progress import Progress, track

import numpy as np
from scipy.ndimage import generic_filter
from concurrent.futures import ProcessPoolExecutor
import time

# cube.shape = (wavel, height, width)
def interpolate_negatives(cube):
    cube = cube.copy()  # Pour ne pas modifier l'original

    def interpolate_func(values):
        center = values[len(values) // 2]
        if center >= 0:
            return center
        else:
            # On garde que les valeurs positives dans le voisinage
            positives = values[values >= 0]
            return np.mean(positives) if len(positives) > 0 else 0.0

    # Appliquer un filtre 3D (voisinage 3x3x3)
    result = generic_filter(cube, interpolate_func, size=3, mode='mirror')
    return result

# Lecture du cube de données
hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Scan/NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits')
data_cube = hdul[1].data
hdr = hdul[1].header
wavel = np.array(hdul[5].data[0])[0,:,0]

# On garde que les canaux 1A, 2ABC, 3ABC et 4AB car le 4C est trop bruité 
wavel = wavel[:9935] # 1A to 4B
raw_data_cube = data_cube[:9935,:,:]


# C'est pas super précis ça, on peut faire mieux ?
""" Rotation du Cube et sélection de la zone d'intérêt pour avoir la zone commune aux 4 canaux """
from scipy.ndimage import rotate
coords = [(36,29), (74,29), (36,75), (74,75)]
angle = -37  # angle en degrés

# Créer un nouveau cube pour stocker les résultats
rotated_cube = np.empty_like(raw_data_cube)
raw_data_cube[np.isnan(raw_data_cube)] = 0 # On fixe les NaN à 0 pour la rotation
# Appliquer la rotation pour chaque tranche spectrale
for i in range(raw_data_cube.shape[0]):
    # Appliquer la rotation en 2D en gardant le centre de la tranche comme centre de rotation
    rotated_cube[i] = rotate(raw_data_cube[i], angle=angle, reshape=False, axes=(1, 0), order=3, mode='nearest')

# Extraire la zone d'intérêt
masked_array = rotated_cube[:, coords[0][1]:coords[2][1], coords[0][0]:coords[3][0]]
spectrum_masked_array = np.nanmean(masked_array, axis=(1,2))


""" Interpolation des valeurs négatives. A enlever si on veut utiliser un offset. """
# raw_data_cube = interpolate_negatives(masked_array)
raw_data_cube = masked_array

### Spectral line processing ###

# On va faire l'étape de détection sur le spectre moyen pour avoir un s/n optimal
mean_spectrum = np.nanmean(raw_data_cube, axis = (1, 2))

# Baseline subtraction
baseline = iterative_baseline_removal(mean_spectrum, lam=1e3, ncycles=5, sigma=1.0)
baseline_subtrated = mean_spectrum - baseline
mad = sliding_mad(baseline_subtrated)

# Peak fitting
fitted_peaks = detect_and_fit_peaks(baseline_subtrated, mad, sigma=5, distance=5)
clean_peaks, mean_sigma, std_sigma = filter_clean_peaks(fitted_peaks)

# là on a une liste de raie spectrales bien détectées sur le spectre moyen, on va s'en servir ensuite
print(clean_peaks)
print(f"Clean peaks: {len(clean_peaks)}")
print(f"Mean sigma (brightest 20%): {mean_sigma:.3f}")
print(f"Std sigma: {std_sigma:.3f}")

peak_indices = [p['peak_index'] for p in clean_peaks]

# clean_peaks : contains the list of robust detections (derived from mean spectrum)
# Now, fit these peaks pixel per pixel
_, I, J = raw_data_cube.shape
print(f"Cube size: {I} x {J} pixels")

results = np.empty((len(peak_indices), I, J), dtype=float)  # optional: store fit results

# New cube: same spectral axis, same spatial size
spectral_line_cube = np.zeros_like(raw_data_cube)
continuum_cube = raw_data_cube.copy()

# On re-boucle, pixel par pixel, et on se sert de la liste de raies spectrales construites plus haut pour faire des fit gaussien aux positions attendues
# Même en connaissant leur position on risque de perdre des raies spectrales qui étaient détectées sur le spectre moyen, car maintenant on fait un fit
# sur des données beaucoup plus bruitées (améliorer avec fit bayésien ?)
# with Progress() as progress:
#     task = progress.add_task("[cyan]Fitting spectra...", total=I*J)

#     for i in range(I):
#         for j in range(J):
#             spectrum = raw_data_cube[:, i, j]
#             baseline_local = iterative_baseline_removal(spectrum, lam=1e3, ncycles=5, sigma=1.0)
#             baseline_subtracted = spectrum - baseline_local

#             # Fit the known peaks
#             fitted, continuum = fit_peaks_only(baseline_subtracted, spectrum, peak_indices, mean_sigma, std_sigma)

#             # Update continuum
#             continuum_cube[:, i, j] = continuum

#             # --- Reconstruct spectrum from fitted Gaussians ---
#             x = np.arange(len(spectrum))
#             for peak in fitted:
#                 A = peak["amplitude"]
#                 mu = peak["center"]
#                 sigma = peak["sigma"]
#                 spectral_line_cube[:, i, j] += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))

#             progress.update(task, advance=1)

# def process_pixel(args):
#     i, j, raw_data_cube, peak_indices, mean_sigma, std_sigma = args
#     spectrum = raw_data_cube[:, i, j]
#     baseline_local = iterative_baseline_removal(spectrum, lam=1e3, ncycles=5, sigma=1.0)
#     baseline_subtracted = spectrum - baseline_local

#     fitted, continuum = fit_peaks_only(baseline_subtracted, spectrum, peak_indices, mean_sigma, std_sigma)

#     x = np.arange(len(spectrum))
#     spectral_line = np.zeros_like(spectrum)
#     for peak in fitted:
#         A = peak["amplitude"]
#         mu = peak["center"]
#         sigma = peak["sigma"]
#         spectral_line += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))

#     return i, j, continuum, spectral_line

# with Progress() as progress:
#     task = progress.add_task("[cyan]Fitting spectra...", total=I * J)

#     continuum_cube = np.zeros_like(raw_data_cube)
#     spectral_line_cube = np.zeros_like(raw_data_cube)

#     args = [(i, j, raw_data_cube, peak_indices, mean_sigma, std_sigma) for i in range(I) for j in range(J)]

#     with ProcessPoolExecutor() as executor:
#         for i, j, continuum, spectral_line in executor.map(process_pixel, args):
#             continuum_cube[:, i, j] = continuum
#             spectral_line_cube[:, i, j] = spectral_line
#             progress.update(task, advance=1)




def process_pixel_chunk(chunk):
    results = []
    for (i, j, raw_data_cube, peak_indices, mean_sigma, std_sigma) in chunk:
        spectrum = raw_data_cube[:, i, j]
        baseline_local = iterative_baseline_removal(spectrum, lam=1e3, ncycles=5, sigma=1.0)
        baseline_subtracted = spectrum - baseline_local
        fitted, continuum = fit_peaks_only(baseline_subtracted, spectrum, peak_indices, mean_sigma, std_sigma)

        x = np.arange(len(spectrum))
        spectral_line = np.zeros_like(spectrum)
        for peak in fitted:
            A, mu, sigma = peak["amplitude"], peak["center"], peak["sigma"]
            spectral_line += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))

        results.append((i, j, continuum, spectral_line))
    return results



CHUNK_SIZE = 46
start = time.time()
with Progress() as progress:
    # task = progress.add_task("[cyan]Fitting spectra...", total=I * J)

    continuum_cube = np.zeros_like(raw_data_cube)
    spectral_line_cube = np.zeros_like(raw_data_cube)

    # Grouping args in chunks
    args = [(i, j, raw_data_cube, peak_indices, mean_sigma, std_sigma) for i in range(I) for j in range(J)]
    chunks = [args[k:k + CHUNK_SIZE] for k in range(0, len(args), CHUNK_SIZE)]

    with ProcessPoolExecutor() as executor:
        for result_chunk in executor.map(process_pixel_chunk, chunks, chunksize=1):
            for i, j, continuum, spectral_line in result_chunk:
                continuum_cube[:, i, j] = continuum
                spectral_line_cube[:, i, j] = spectral_line
                # progress.advance(task, 1)

end = time.time()
print(f"Chunk size: {CHUNK_SIZE}, Time taken: {end - start:.2f} seconds")









# np.save("/home/pdellova/spectral_line_cube.npy", spectral_line_cube)
# np.save("/home/pdellova/continuum_cube.npy", continuum_cube)
#spectral_line_cube = np.load("/home/pdellova/spectral_line_cube.npy")
#continuum_cube = np.load("/home/pdellova/continuum_cube.npy")

_, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# Top panel: continuum vs. baseline
axs[0].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)) + np.nanmean(spectral_line_cube, axis=(1, 2)), label='Separated Continuum', color='red', alpha = 0.75)
axs[0].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)), label='Input Spectrum', color='black', alpha=0.75)
axs[0].legend()

# --- Bottom panel: Gaussian fits over extracted lines ---
axs[1].plot(wavel, np.nanmean(raw_data_cube, axis=(1, 2)) - baseline, color='black', label='Separated lines', alpha=0.75)
axs[1].plot(wavel, np.nanmean(spectral_line_cube, axis=(1, 2)), color='teal', label='Gaussians (best fit)', alpha=0.75)
axs[1].plot(wavel, np.nanmean(raw_data_cube, axis=(1, 2)) - baseline - np.nanmean(spectral_line_cube, axis=(1, 2)) - 5e2, color='crimson', label='Residuals (shifted)', alpha=0.75)
axs[1].set_xlabel("Wavelength (microns)")
axs[1].legend()

plt.tight_layout()
# plt.show()

continuum_data_cube = continuum_cube

# --- Negative values processing --- #

# Try spatial median filtering to reduce negative pixels
filter_continuum_data_cube = np.empty_like(continuum_data_cube)
for k in range(continuum_data_cube.shape[0]):
    layer = continuum_data_cube[k]
    med = ndimage.median_filter(layer, size=3)  # noyau 3x3 spatial
    # On remplace uniquement les pixels négatifs par la médiane locale
    mask_neg = layer < 0
    layer_corrected = layer.copy()
    layer_corrected[mask_neg] = med[mask_neg]
    filter_continuum_data_cube[k] = layer_corrected

# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None
print(f"Negative value analysis after Spatial median filtering of size 3x3:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")


print("---------------")


# First Try with global median filter to reduce negative pixels 
filter_continuum_data_cube = ndimage.median_filter(continuum_data_cube, size=3, axes=[0])

# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Negative value analysis after median filtering of size 3:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")


print("---------------")


filter_continuum_data_cube = ndimage.median_filter(continuum_data_cube, size=5, axes=[0])

# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Negative value analysis after median filtering of size 5:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")

print("---------------")


filter_continuum_data_cube = ndimage.median_filter(continuum_data_cube, size=7, axes=[0])

# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Negative value analysis after median filtering of size 7:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")
print("---------------")


# Now Try with pixel median filter to reduce negative pixels 
continuum_data_cube = continuum_data_cube.astype(np.float64)
filter_continuum_data_cube = np.zeros_like(continuum_data_cube)
print(continuum_data_cube.shape, continuum_data_cube.dtype)
for i in range(continuum_data_cube.shape[1]):
    for j in range(continuum_data_cube.shape[2]):
        # pixel_spectrum = continuum_data_cube[:, i, j]
        # print(type(pixel_spectrum), pixel_spectrum.shape)
        filter_continuum_data_cube[:, i, j] = ndimage.median_filter(continuum_data_cube[:, i, j], size=3)
        # filter_continuum_data_cube[:, i, j] = pixel_spectrum
# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Negative value analysis after Pixel to pixel median filtering of size 3:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")
print("---------------")
  

filter_continuum_data_cube = np.zeros_like(continuum_data_cube)
for i in range(continuum_data_cube.shape[1]):
    for j in range(continuum_data_cube.shape[2]):
        pixel_spectrum = continuum_data_cube[:, i, j]
        median_value = ndimage.median_filter(pixel_spectrum, size=5)
        filter_continuum_data_cube[:, i, j] = pixel_spectrum
# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Negative value analysis after median filtering of size 5:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")
  
print("---------------")

filter_continuum_data_cube = np.zeros_like(continuum_data_cube)
for i in range(continuum_data_cube.shape[1]):
    for j in range(continuum_data_cube.shape[2]):
        pixel_spectrum = continuum_data_cube[:, i, j]
        median_value = ndimage.median_filter(pixel_spectrum, size=7)
        filter_continuum_data_cube[:, i, j] = pixel_spectrum
# Identify negatives
negatives = filter_continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = filter_continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = filter_continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Negative value analysis after median filtering of size 7:")
print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")

raise SystemExit
""" Offset du continium pour éviter les valeurs négatives """
continuum_data_cube += 2*np.abs(min_negative)

# Need to set negative values to zero for NMF (due to line subtraction in low s/n channels)
# continuum_data_cube[continuum_data_cube < 0] = 0
# spectral_line_spectrum = np.nanmean(spectral_line_cube, axis=(1, 2))

# raise SystemExit

### Component separation ###

""" Changement de shape des données pour appliquer la NMF """
masked_array_fitlered_data = rearrange(continuum_data_cube, 'L I J -> (I J) L') # from spectro data


""" Application de la NMF avec le nombre de composantes choisi """
# Initialize NMF with the desired number of components
nmf = NMF(n_components=6, init='random', random_state=0, max_iter=1000)
nmf.fit(masked_array_fitlered_data) # Fit NMF model to your data
components = nmf.components_ # Extract the components (eigenvectors)

n_components = components.shape[0]  # ici 6
n_components -= 2*np.abs(min_negative)

# On ajoute les raies spectrale dans les composantes à sauvegarder
for peak in range(len(peak_indices)):
    peak_line = np.zeros(components.shape[1])
    peak_line[peak_indices[peak]] = mean_spectrum[peak_indices[peak]]
    components = np.vstack([components, peak_line])

# continuum_data_cube[continuum_data_cube==0] = np.nan
# raw_data_cube[raw_data_cube==0] = np.nan

# np.save(f'/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_{components.shape[0]}_templates.npy', components)
