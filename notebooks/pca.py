import numpy as np
from astropy.io import fits
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import ndimage

from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

from einops import rearrange
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

from utils import iterative_baseline_removal, detect_and_fit_peaks, sliding_mad, filter_clean_peaks, build_line_spectrum, build_line_spectrum_from_gaussians, fit_peaks_only
from rich.progress import Progress, track
from numba import njit, prange
import numpy as np
from scipy.ndimage import generic_filter
from concurrent.futures import ProcessPoolExecutor


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


def process_pixel(args):
    i, j, raw_data_cube, peak_indices, mean_sigma, std_sigma = args
    spectrum = raw_data_cube[:, i, j]
    baseline_local = iterative_baseline_removal(spectrum, lam=1e3, ncycles=5, sigma=1.0)
    baseline_subtracted = spectrum - baseline_local

    fitted, continuum = fit_peaks_only(baseline_subtracted, spectrum, peak_indices, mean_sigma, std_sigma)

    x = np.arange(len(spectrum))
    spectral_line = np.zeros_like(spectrum)
    for peak in fitted:
        A = peak["amplitude"]
        mu = peak["center"]
        sigma = peak["sigma"]
        spectral_line += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))

    return i, j, continuum, spectral_line

with Progress() as progress:
    task = progress.add_task("[cyan]Fitting spectra...", total=I * J)

    continuum_cube = np.zeros_like(raw_data_cube)
    spectral_line_cube = np.zeros_like(raw_data_cube)

    args = [(i, j, raw_data_cube, peak_indices, mean_sigma, std_sigma) for i in range(I) for j in range(J)]

    with ProcessPoolExecutor() as executor:
        for i, j, continuum, spectral_line in executor.map(process_pixel, args):
            continuum_cube[:, i, j] = continuum
            spectral_line_cube[:, i, j] = spectral_line
            progress.update(task, advance=1)

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
plt.show()

continuum_data_cube = continuum_cube

# --- Negative values processing --- #

# Identify negatives
negatives = continuum_data_cube < 0
num_negatives = np.sum(negatives)
total_pixels = continuum_data_cube.size
fraction_negatives = num_negatives / total_pixels

# Amplitude stats of negatives
negative_values = continuum_data_cube[negatives]
min_negative = negative_values.min() if num_negatives > 0 else None
mean_negative = negative_values.mean() if num_negatives > 0 else None
max_negative = negative_values.max() if num_negatives > 0 else None

print(f"Total pixels: {total_pixels:,}")
print(f"Negative pixels: {num_negatives:,} ({fraction_negatives:.4%})")
if num_negatives > 0:
    print(f"Negative amplitude (min/mean/max): {min_negative:.3e} / {mean_negative:.3e} / {max_negative:.3e}")


""" Offset du continium pour éviter les valeurs négatives """
# continuum_data_cube += 2*np.abs(min_negative)

# Need to set negative values to zero for NMF (due to line subtraction in low s/n channels)
# continuum_data_cube[continuum_data_cube < 0] = 0
# spectral_line_spectrum = np.nanmean(spectral_line_cube, axis=(1, 2))

# raise SystemExit

### Component separation ###

""" Changement de shape des données pour appliquer la NMF """


spectro = continuum_data_cube.reshape(continuum_data_cube.shape[0],continuum_data_cube.shape[1]*continuum_data_cube.shape[2])
nan_mask = np.isnan(spectro)

# Find the indices of non-NaN values
indices = ndimage.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)

# Replace NaNs with nearest non-NaN values using the indices
filled_image = spectro[tuple(indices)]

pca = PCA(n_components=6)
principal_components = pca.fit_transform(filled_image)

principal_components = principal_components.T

plt.figure()
for i in range(principal_components.shape[0]):
    plt.plot(wavel, principal_components[i], label=i)
plt.show()

n_components = principal_components.shape[0]  # ici 6

# On ajoute les raies spectrale dans les composantes à sauvegarder
for peak in range(len(peak_indices)):
    peak_line = np.zeros(principal_components.shape[1])
    peak_line[peak_indices[peak]] = mean_spectrum[peak_indices[peak]]
    principal_components = np.vstack([principal_components, peak_line])

# continuum_data_cube[continuum_data_cube==0] = np.nan
# raw_data_cube[raw_data_cube==0] = np.nan

np.save(f'/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/pca_NGC7023_1ABC_2ABC_3ABC_4AB_{principal_components.shape[0]}_templates.npy', principal_components)
