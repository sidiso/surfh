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
thin_wavel = np.array(hdul[5].data[0])[0,:,0]

# On garde que les canaux 1A, 2ABC, 3ABC et 4AB car le 4C est trop bruité 
wavel = wavel[:9935] # 1A to 4B
raw_data_cube = data_cube[:9935,:,:]
thin_wavel_cut = thin_wavel[:9935]


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
fitted_peaks = detect_and_fit_peaks(baseline_subtrated, mad, sigma=5, distance=5, wavelength=wavel)
clean_peaks, mean_sigma, std_sigma = filter_clean_peaks(fitted_peaks)

# là on a une liste de raie spectrales bien détectées sur le spectre moyen, on va s'en servir ensuite
print(clean_peaks)
print(f"Clean peaks: {len(clean_peaks)}")
print(f"Mean sigma (brightest 20%): {mean_sigma:.3f}")
print(f"Std sigma: {std_sigma:.3f}")

peak_indices = [p['peak_index'] for p in clean_peaks]

# On considère une linéarité entre deux points de longueurs d'onde
# On traite les peak par ordre décroissant de wavelength pour éviter les problèmes d'insertion
for peak in sorted(clean_peaks, key=lambda p: p['center'], reverse=True):    
    centroid = peak['center']
    # We interpolate the value
    i0 = int(np.floor(centroid))
    i1 = int(np.ceil(centroid))
    frac = centroid - i0
    wavelength_center = wavel[i0] * (1 - frac) + wavel[i1] * frac

    thin_wavel = np.insert(thin_wavel, i1, wavelength_center)
    thin_wavel_cut = np.insert(thin_wavel_cut, i1, wavelength_center)

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

_, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# Top panel: continuum vs. baseline
axs[0].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)) + np.nanmean(spectral_line_cube, axis=(1, 2)), label='Separated Continuum', color='teal', alpha = 0.75)
axs[0].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)), label='Input Spectrum', color='black', linestyle='dashed',alpha=0.75)
axs[1].set_ylabel(r"Flux (MJy sr$^{-1}$)")
axs[0].legend()

# --- Bottom panel: Gaussian fits over extracted lines ---
axs[1].plot(wavel, np.nanmean(raw_data_cube, axis=(1, 2)) - baseline, color='black', linestyle='dashed', label='Separated lines', alpha=0.75)
axs[1].plot(wavel, np.nanmean(spectral_line_cube, axis=(1, 2)), color='teal', label='Gaussians (best fit)', alpha=0.75)
axs[1].plot(wavel, np.nanmean(raw_data_cube, axis=(1, 2)) - baseline - np.nanmean(spectral_line_cube, axis=(1, 2)) - 5e2, color='crimson', linestyle='dotted', label='Residuals (shifted)', alpha=0.75)
axs[1].set_xlabel("Wavelength (microns)")
axs[1].set_ylabel(r"Flux (MJy sr$^{-1}$)")
axs[1].legend()

plt.tight_layout()



# # --- Visualization of results --- #
# _, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

# # Top panel: continuum vs. baseline
# axs[0].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)) + np.nanmean(spectral_line_cube, axis=(1, 2)), label='Separated Continuum', color='red', alpha = 0.75)
# axs[0].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)), label='Input Spectrum', color='black', alpha=0.75)
# axs[0].legend()

# # --- Bottom panel: Gaussian fits over extracted lines ---
# axs[1].plot(wavel, np.nanmean(raw_data_cube, axis=(1, 2)) - baseline, color='black', label='Separated lines', alpha=0.75)
# axs[1].plot(wavel, np.nanmean(spectral_line_cube, axis=(1, 2)), color='teal', label='Gaussians (best fit)', alpha=0.75)
# axs[1].plot(wavel, np.nanmean(raw_data_cube, axis=(1, 2)) - baseline - np.nanmean(spectral_line_cube, axis=(1, 2)) - 5e2, color='crimson', label='Residuals (shifted)', alpha=0.75)
# axs[1].set_xlabel("Wavelength (microns)")
# axs[1].legend()

# plt.tight_layout()
plt.show()

continuum_data_cube = continuum_cube
# --- Negative values processing --- #
# On commence par mettre à NaN les valeurs négatives 
continuum_data_cube = np.where(continuum_data_cube < 0, continuum_data_cube, continuum_data_cube)

from scipy.signal import medfilt

# Taille du filtre sur la dimension spectrale 
kernel_size = 5
print("Filtering Cube")
# Appliquer le filtre médian le long du premier axe (lambda)
filtered_cube = np.empty_like(continuum_data_cube)
for y in range(continuum_data_cube.shape[1]):
    for x in range(continuum_data_cube.shape[2]):
        spectrum = continuum_data_cube[:, y, x]
        # Remplacer les NaN temporairement pour filtrer
        temp = np.nan_to_num(spectrum, nan=np.nanmedian(spectrum))
        filtered_cube[:, y, x] = medfilt(temp, kernel_size=kernel_size)
print("Filtering done")

filtered_cube = np.where(filtered_cube < 0, np.nan, filtered_cube)


from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
cube = filtered_cube  


# Masque des valeurs valides
mask = np.isfinite(cube)
cube_filled = np.nan_to_num(cube, nan=0.0)

# Rayon de lissage (lmanda, y, x)
sigma = (1, 2, 2)  

# Moyenne pondérée par le masque
print("Smoothing Cube")
smoothed_data = gaussian_filter(cube_filled, sigma=sigma, mode='nearest')
smoothed_weights = gaussian_filter(mask.astype(float), sigma=sigma, mode='nearest')

# Reconstruction : données lissées / poids
cube_interp = smoothed_data / smoothed_weights

# On re
cube_interp[~np.isfinite(cube_interp)] = 0.0
print("Smoothing done")
print("Check number of 0 values after interpolation:", np.sum(cube_interp == 0))

# Check si tout est positif
cube_interp = np.where(cube_interp < 0, 0, cube_interp)

# Top panel: continuum vs. baseline
axs[2].plot(wavel, np.nanmean(cube_filled, axis=(1,2)), label='Filtered Continiuum', color='red', alpha = 0.75)
axs[2].plot(wavel, np.nanmean(continuum_cube, axis=(1, 2)), label='Continiuum ', color='black', alpha=0.75)
axs[2].legend()
# plt.show()

# raise SystemExit
""" Offset du continium pour éviter les valeurs négatives """

# Need to set negative values to zero for NMF (due to line subtraction in low s/n channels)
# continuum_data_cube[continuum_data_cube < 0] = 0
# spectral_line_spectrum = np.nanmean(spectral_line_cube, axis=(1, 2))

# raise SystemExit

### Component separation ###

""" Changement de shape des données pour appliquer la NMF """
# masked_array_fitlered_data = rearrange(continuum_data_cube, 'L I J -> (I J) L') # from spectro data
masked_array_fitlered_data = rearrange(cube_interp, 'L I J -> (I J) L') # from spectro data



""" Application de la NMF avec le nombre de composantes choisi """
# Initialize NMF with the desired number of components
nmf = NMF(n_components=6, init='random', random_state=0, max_iter=1000)
nmf.fit(masked_array_fitlered_data) # Fit NMF model to your data
components = nmf.components_ # Extract the components (eigenvectors)

# On interpole les composantes sur les longueurs d'onde avec le rajout des raies spectrale
thin_components = np.zeros((components.shape[0], len(thin_wavel_cut)))
for i in range(components.shape[0]):
    thin_components[i] = np.interp(thin_wavel_cut, wavel, components[i])

# On ajoute les raies spectrale dans les composantes à sauvegarder
for peak in range(len(peak_indices)):
    peak_line = np.zeros(thin_components.shape[1])
    print(clean_peaks[peak]['wavel_center'])
    idx = np.argmin(np.abs(thin_wavel_cut-clean_peaks[peak]['wavel_center']))
    print(idx)
    peak_line[idx] = mean_spectrum[peak_indices[peak]]
    thin_components = np.vstack([thin_components, peak_line])




# continuum_data_cube[continuum_data_cube==0] = np.nan
# raw_data_cube[raw_data_cube==0] = np.nan

print(f"Shape before {wavel.shape}, After {thin_wavel_cut.shape}")
# np.save(f'/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_{components.shape[0]}_templates.npy', components)
np.save(f'/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_{thin_components.shape[0]}_templates.npy', thin_components)
np.save(f'/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy', thin_wavel_cut)
np.save(f'/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy', thin_wavel)