import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate
from skimage.draw import polygon

import scipy
import time 

from surfh.ToolsDir.alignment import interactive_align
from surfh.Algorithm.alignment_correction import AlignmentCorrection
from surfh.Signalprocessing import baseline, fitting, line_extraction, model, utilities




ref_path = '/home/nmonnier/Data/JWST/NGC_7023/Scan/'
# fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_204_mu_5.00e+06_SD_True/'
# fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_1000_mu_5.00e+06_SD_True/'
# fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_800_mu_5.00e+06_SD_True/'
fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_801_mu_5.00e+06_SD_True/'


# Load Templates of shape (N, n_lambda) to get position of spectrale line in fusion data
templates = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_16_templates.npy')
sum_lines = np.zeros((templates.shape[1]))
for i in range(templates.shape[0]): # Car il y a 8 spectral lines dans les templates
    if np.sum(templates[i]!=0) <10:
        sum_lines += templates[i]

nonlines_idx = np.where(sum_lines == 0)[0]




# open reference FITS file
ref_file = ref_path + 'NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits'
with fits.open(ref_file) as ref_hdu:
    ref_data = ref_hdu[1].data
    ref_header = ref_hdu[1].header
ref_data[np.isnan(ref_data)] = 0

# open fusion result FITS file
fusion_file = fusion_path + 'res_cube.fits'
with fits.open(fusion_file) as fusion_hdu:
    fusion_data = fusion_hdu[0].data
    fusion_header = fusion_hdu[0].header
    wavelength = fusion_hdu['WCS-TABLE'].data['wavelength'][0]
    wavelength = wavelength.squeeze() # Because shape can be (N,1) or (1,N)
    PA = fusion_header['PA_V3']

raw_wavelength_length = len(nonlines_idx)

# Load masks and set masks wzvelength range
try:
    with fits.open(fusion_file) as fusion_hdu:
        masks = fusion_hdu['MASKS'].data
        print(f"Masks shape = {masks.shape}")
except:
    raise KeyError
# masks = np.load(fusion_path + 'masks.npy')
ch_limit = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 24.19]

# Pad spatial dimension of ref data to be the same shape as spacial dimension of fusion data
if ref_data.shape[1] < fusion_data.shape[1] or ref_data.shape[2] < fusion_data.shape[2]:
    pad_y = (fusion_data.shape[1] - ref_data.shape[1]) // 2
    pad_x = (fusion_data.shape[2] - ref_data.shape[2]) // 2
    ref_data = np.pad(ref_data, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    print(f"Padded ref data to shape: {ref_data.shape}")

sum_fusion_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
sum_ref_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
print(f"SHape ref data: {sum_fusion_cube.shape}, fusion data: {fusion_data.shape}")

# Apply masks to dedicated wavelength 
i = 0
idx_prev = 0 
for wave in ch_limit:
    idx_wavel = np.argmin(np.abs(wavelength - wave))
    fusion_data[idx_prev:idx_wavel] = fusion_data[idx_prev:idx_wavel,:,:]*masks[i]
    idx_prev = idx_wavel
    i +=1

# Build summed cubes over each channel to perform alignement correction
sum_fusion_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
sum_ref_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
i = 0
idx_prev = 0 
for wave in ch_limit:
    idx_wavel = np.argmin(np.abs(wavelength - wave))
    print(f"Band {i}: Wavelength {wave} um, Index range: {idx_prev} to {idx_wavel}")
    sum_fusion_cube[i] = np.nansum(fusion_data[idx_prev:idx_wavel,:,:], axis=0)
    sum_ref_cube[i] = np.nansum(ref_data[idx_prev:idx_wavel], axis=0)
    idx_prev = idx_wavel
    i +=1


# Select specific band to perform alignement
band_idx = 6
fusion_slice = rotate(sum_fusion_cube[band_idx], angle=PA-180, reshape=False) # Here rotation is needed due to JWST convention
ref_slice = sum_ref_cube[band_idx]


norm_fusion_data = fusion_slice/np.max(fusion_slice)
norm_ref_data = ref_slice/np.max(ref_slice)

corrected_fusion_data, optimal_params = AlignmentCorrection.alignement_correction(
                                                                            AlignmentCorrection.chi2, 
                                                                            norm_ref_data,
                                                                            norm_fusion_data,
                                                                            initial_guess=(0,0,0)
                                                                            )

rotation_opt, y_shift_opt, x_shift_opt = optimal_params.x
print(f"Optimal parameters: rotation = {rotation_opt}, y_shift = {y_shift_opt}, x_shift = {x_shift_opt}")


def ensure_native_endian(arr):
    if arr.dtype.byteorder not in ('=', '<'):
        arr = arr.byteswap().newbyteorder()
    return arr

fusion_data = ensure_native_endian(fusion_data)

start = time.time()
rotated_fusion_cube = AlignmentCorrection.apply_cube_shift_rotation(fusion_data, (PA-180) + rotation_opt, y_shift_opt, x_shift_opt)
end = time.time()
print(f"Time taken to apply correction to the entire cube: {end - start} seconds")


# Build summed cubes over each channel to perform alignement correction
sum_fusion_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
sum_ref_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
i = 0
idx_prev = 0 
for wave in ch_limit:
    idx_wavel = np.argmin(np.abs(wavelength - wave))
    print(f"Band {i}: Wavelength {wave} um, Index range: {idx_prev} to {idx_wavel}")
    sum_fusion_cube[i] = np.nansum(rotated_fusion_cube[idx_prev:idx_wavel,:,:], axis=0)
    sum_ref_cube[i] = np.nansum(ref_data[idx_prev:idx_wavel], axis=0)
    idx_prev = idx_wavel
    i +=1


# interactive_align(sum_ref_cube[0], sum_fusion_cube[0], np.ones_like(corrected_fusion_data))


""" Select common FoV for all wavelength """
def extract_region_signal(cube, vertices):
    """
    Extrait le signal moyen (ou individuel) pour une zone polygonale définie
    par 4 points dans un cube 3D (n_lambda, height, width).

    Paramètres
    ----------
    cube : np.ndarray
        Cube 3D (n_lambda, height, width).
    vertices : list of tuple
        Liste de points [(y1, x1), (y2, x2), (y3, x3), (y4, x4)]
        définissant un polygone fermé (pas forcément droit).

    Retourne
    --------
    region_signals : np.ndarray
        Tableau 2D de forme (n_lambda, n_pixels) contenant les spectres de chaque pixel de la région.
    mean_signal : np.ndarray
        Spectre moyen (moyenne sur tous les pixels de la région).
    mask : np.ndarray
        Masque binaire 2D (True à l’intérieur du polygone).
    """
    # Coordonnées du polygone
    r = np.array([v[0] for v in vertices])
    c = np.array([v[1] for v in vertices])
    
    # Pixels à l’intérieur du polygone
    rr, cc = polygon(r, c, cube.shape[1:])
    mask = np.zeros(cube.shape[1:], dtype=bool)
    mask[rr, cc] = True
    
    return mask


points = [(54, 30), (35, 57), (67, 81), (88, 58)]  # (y, x)
mask = extract_region_signal(rotated_fusion_cube, points)

region_fusion_cube = rotated_fusion_cube[:, mask]
region_raw_cube = ref_data[:, mask]

mean_fusion_cube = np.mean(region_fusion_cube, axis=1)
print("raw_wavelength_length = ", raw_wavelength_length)
mean_raw_cube = np.mean(region_raw_cube, axis=1)[:raw_wavelength_length]


""" Process the mean spectra to extract spectral lines on raw data"""
baseline_raw = baseline.iterative_baseline_removal(mean_raw_cube, lam=1e3, ncycles=5, sigma=1.0)
baseline_subtracted = mean_raw_cube - baseline_raw
mad = utilities.sliding_mad(baseline_subtracted)
# Peak fitting
fitted_peaks = fitting.detect_and_fit_peaks(baseline_subtracted, mad, sigma=5, distance=5)
clean_peaks, mean_sigma, std_sigma = fitting.filter_clean_peaks(fitted_peaks)
peak_indices = [p['peak_index'] for p in clean_peaks]

fitted, continuum_raw = fitting.fit_peaks_only(baseline_subtracted, mean_raw_cube, peak_indices, mean_sigma, std_sigma)

x = np.arange(len(mean_raw_cube))
spectral_line_raw = np.zeros_like(mean_raw_cube)
for peak in fitted:
    A, mu, sigma = peak["amplitude"], peak["center"], peak["sigma"]
    spectral_line_raw += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))


""" Process the mean spectra to extract spectral lines on fusion data"""
baseline_fusion = baseline.iterative_baseline_removal(mean_fusion_cube, lam=1e3, ncycles=5, sigma=1.0)
baseline_subtracted = mean_fusion_cube - baseline_fusion
mad = utilities.sliding_mad(baseline_subtracted)
# Peak fitting
fitted_peaks = fitting.detect_and_fit_peaks(baseline_subtracted, mad, sigma=5, distance=5)
clean_peaks, mean_sigma, std_sigma = fitting.filter_clean_peaks(fitted_peaks)
peak_indices = [p['peak_index'] for p in clean_peaks]

# fitted, continuum_fusion = fitting.fit_peaks_only(baseline_subtracted, mean_raw_cube, peak_indices, mean_sigma, std_sigma)

# x = np.arange(len(mean_raw_cube))
# spectral_line_fusion = np.zeros_like(mean_fusion_cube)
# for peak in fitted:
#     A, mu, sigma = peak["amplitude"], peak["center"], peak["sigma"]
#     spectral_line_fusion += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))





""" Filter the continuum cube to remove remaining artefacts """
continuum_data_cube = ref_data
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
mask_inf = np.isfinite(cube)
cube_filled = np.nan_to_num(cube, nan=0.0)
# Rayon de lissage (lmanda, y, x)
sigma = (1, 2, 2)  
# Moyenne pondérée par le masque
print("Smoothing Cube")
smoothed_data = gaussian_filter(cube_filled, sigma=sigma, mode='nearest')
smoothed_weights = gaussian_filter(mask_inf.astype(float), sigma=sigma, mode='nearest')
# Reconstruction : données lissées / poids
cube_interp = smoothed_data / smoothed_weights
# On re
cube_interp[~np.isfinite(cube_interp)] = 0.0
print("Smoothing done")
print("Check number of 0 values after interpolation:", np.sum(cube_interp == 0))
# Check si tout est positif
cube_interp = np.where(cube_interp < 0, 0, cube_interp)

region_cube_interp = cube_interp[:, mask]

mean_cube_interp = np.mean(region_cube_interp, axis=1)[:raw_wavelength_length]


""" Process the mean spectra to extract spectral lines on raw data"""
baseline_cube_interp = baseline.iterative_baseline_removal(mean_cube_interp, lam=1e3, ncycles=5, sigma=1.0)
baseline_subtracted = mean_cube_interp - baseline_cube_interp
mad = utilities.sliding_mad(baseline_subtracted)
# Peak fitting
fitted_peaks = fitting.detect_and_fit_peaks(baseline_subtracted, mad, sigma=5, distance=5)
clean_peaks, mean_sigma, std_sigma = fitting.filter_clean_peaks(fitted_peaks)
peak_indices = [p['peak_index'] for p in clean_peaks]

fitted, continuum_cube_interp = fitting.fit_peaks_only(baseline_subtracted, mean_cube_interp, peak_indices, mean_sigma, std_sigma)

x = np.arange(len(mean_cube_interp))
spectral_line_cube_interp = np.zeros_like(mean_cube_interp)
for peak in fitted:
    A, mu, sigma = peak["amplitude"], peak["center"], peak["sigma"]
    spectral_line_cube_interp += A * np.exp(-(x - mu) ** 2 / (2 * sigma**2))


# Load Templates of shape (N, n_lambda) to get position of spectrale line in fusion data
templates = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_16_templates.npy')
sum_lines = np.zeros((templates.shape[1]))
continium_fusion_cube = np.copy(mean_fusion_cube)
for i in range(templates.shape[0]): # Car il y a 8 spectral lines dans les templates
    # Check if two consecutive values are equal to zero
    # if np.any(np.diff(templates[i] == 0) == 0):
        # continue
    idx = np.argmin(np.abs(templates[i]  - np.max(templates[i])) )
    continium_fusion_cube[idx] = (mean_fusion_cube[idx-1] + mean_fusion_cube[idx+1])/2

    if np.sum(templates[i]!=0) <10:
        sum_lines += templates[i]

nonlines_idx = np.where(sum_lines == 0)[0]


# # Plot The two signals in the same plot
# _, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
# axes[0].plot(wavelength, mean_fusion_cube, label='Fusion Data', color='blue', drawstyle='steps-mid')
# axes[0].plot(wavelength, mean_raw_cube[:len(wavelength)], label='Reference Data', color='red', drawstyle='steps-mid')
# axes[0].set_xlabel('Wavelength (um)')
# axes[0].set_title('Mean Spectrum in Selected Region')
# axes[0].legend()
# axes[0].grid()
# # Plot relative difference between the two signals
# relative_difference = (continium_fusion_cube - continuum_raw) / continuum_raw
# axes[1].plot(wavelength, relative_difference, label='Relative Difference', color='green', drawstyle='steps-mid')
# axes[1].set_xlabel('Wavelength (um)')
# axes[1].set_title('Relative Difference between Fusion and Reference Data')
# axes[1].axhline(0, color='black', linestyle='--')
# axes[1].legend()

# filtered_relative_difference = (continium_fusion_cube - continuum_cube_interp) /continuum_cube_interp
# axes[2].plot(wavelength, filtered_relative_difference, label='Relative Difference', color='green', drawstyle='steps-mid')
# axes[2].set_xlabel('Wavelength (um)')
# axes[2].set_title('Relative Difference between Fusion and Reference Data')
# axes[2].axhline(0, color='black', linestyle='--')
# axes[2].legend()

# axes[3].plot(wavelength, continium_fusion_cube , label='continium_fusion_cube', color='purple', drawstyle='steps-mid')
# axes[3].axhline(0, color='black', linestyle='--')


""" Plot Raw Continiuum + fusion continiuum + Raw spectral lines"""
# _, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
# # Top panel: continuum vs. baseline
# axs[0].plot(wavelength, continuum_raw + spectral_line_raw, label='Separated Continuum', color='red', alpha = 0.75, drawstyle='steps-mid')
# axs[0].plot(wavelength, continuum_raw, label='Input Spectrum', color='black', alpha=0.75, drawstyle='steps-mid')
# axs[0].plot(wavelength, continuum_cube_interp, label='Filtered Continuum', color='blue', alpha=0.75, drawstyle='steps-mid')
# axs[0].legend()
# # --- Bottom panel: Gaussian fits over extracted lines ---
# axs[1].plot(wavelength, mean_raw_cube - baseline_raw, color='black', label='Separated lines', alpha=0.75, drawstyle='steps-mid')
# axs[1].plot(wavelength,spectral_line_raw, color='teal', label='Gaussians (best fit)', alpha=0.75, drawstyle='steps-mid')
# axs[1].plot(wavelength, mean_raw_cube - baseline_raw - spectral_line_raw  - 5e2, color='crimson', label='Residuals (shifted)', alpha=0.75, drawstyle='steps-mid')
# axs[1].set_xlabel("Wavelength (microns)")
# axs[1].legend()


# plt.tight_layout()




# _, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
# # Top panel: continuum vs. baseline
# axs[0].plot(wavelength, continuum_fusion + spectral_line_fusion, label='Separated Continuum', color='red', alpha = 0.75)
# axs[0].plot(wavelength, continuum_fusion, label='Input Spectrum', color='black', alpha=0.75)
# axs[0].legend()
# # --- Bottom panel: Gaussian fits over extracted lines ---
# axs[1].plot(wavelength, mean_fusion_cube - baseline_fusion, color='black', label='Separated lines', alpha=0.75)
# axs[1].plot(wavelength,spectral_line_fusion, color='teal', label='Gaussians (best fit)', alpha=0.75)
# axs[1].plot(wavelength, mean_fusion_cube - baseline_fusion - spectral_line_fusion  - 5e2, color='crimson', label='Residuals (shifted)', alpha=0.75)
# axs[1].set_xlabel("Wavelength (microns)")
# axs[1].legend()


# # axs[2].plot(wavelength, mean_fusion_cube, color='black', label='mean_fusion_cube', alpha=0.75)
# axs[2].plot(wavelength,continium_fusion_cube, color='teal', label='continium_fusion_cube', alpha=0.75)
# axs[2].set_xlabel("Wavelength (microns)")
# axs[2].legend()
# plt.tight_layout()




# Créer la figure avec GridSpec
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2])  # 2 colonnes, gauche fine, droite large

val = float(wavelength[50])
val_fmt = f"{val:.4g}"   # 2 chiffres significatifs


""" Figure 1 : Images et spectres moyens dans la région sélectionnée """
poly = points + [points[0]]
# Images à gauche
ax_img1 = fig.add_subplot(gs[0,0])
ax_img2 = fig.add_subplot(gs[1,0])
ax_img1.imshow(rotated_fusion_cube[50], cmap='viridis')
ax_img1.set_title(fr"Fusion data — $\lambda = {val_fmt}$ um")
ax_img1.plot(*zip(*[(x, y) for y, x in poly]), color="red", lw=1.5)
ax_img2.imshow(ref_data[50], cmap='viridis')
ax_img2.set_title(fr"Raw data — $\lambda = {val_fmt}$ um")
ax_img2.plot(*zip(*[(x, y) for y, x in poly]), color="red", lw=1.5)
# Signal moyen à droite
ax_signal = fig.add_subplot(gs[:,1])  # occupe les deux lignes de la colonne de droite
print(f"wavelength shape = {wavelength.shape}, mean_fusion_cube shape = {mean_fusion_cube.shape}")
ax_signal.plot(wavelength, mean_fusion_cube, label='Mean flux Fusion')
print(f"wavelength shape = {wavelength.shape}, mean_raw_cube shape = {mean_raw_cube.shape}")
ax_signal.plot(wavelength[nonlines_idx], mean_raw_cube, label='Mean flux Raw')
ax_signal.plot(wavelength[nonlines_idx], continuum_cube_interp, label='Filtered Continuum Raw', linestyle='--')
ax_signal.set_xlabel('Wavelength (µm)', fontsize=14)
ax_signal.set_ylabel('Flux', fontsize=14)
ax_signal.set_title('Mean Spectrum in Selected Region', fontsize=16)
ax_signal.legend(fontsize=9)
ax_signal.grid(True)

plt.tight_layout()


fig2 = plt.figure(figsize=(15, 6))
gs2 = fig2.add_gridspec(2, 2, width_ratios=[1, 2])  # 2 colonnes, gauche fine, droite large

poly = points + [points[0]]
# Images à gauche
ax_img1_2 = fig2.add_subplot(gs2[0,0])
ax_img2_2 = fig2.add_subplot(gs2[1,0])
ax_img1_2.imshow(rotated_fusion_cube[50], cmap='viridis')
ax_img1_2.set_title(fr"Fusion data — $\lambda = {val_fmt}$ um")
ax_img1_2.plot(*zip(*[(x, y) for y, x in poly]), color="red", lw=1.5)
ax_img2_2.imshow(ref_data[50], cmap='viridis')
ax_img2_2.set_title(fr"Raw data — $\lambda = {val_fmt}$ um")
ax_img2_2.plot(*zip(*[(x, y) for y, x in poly]), color="red", lw=1.5)
# Signal moyen à droite
relative_difference = (continium_fusion_cube[nonlines_idx] - continuum_raw) / continuum_raw
print(f"Mean relative difference = {np.mean(relative_difference)}")
ax_signal_2 = fig2.add_subplot(gs2[0,1])  # occupe les deux lignes de la colonne de droite
ax_signal_2.plot(wavelength[nonlines_idx], relative_difference, label='Relative Di    rence', color='green', drawstyle='steps-mid')
ax_signal_2.set_xlabel('Wavelength (µm)', fontsize=14)
ax_signal_2.set_ylabel('Flux', fontsize=14)
ax_signal_2.set_title('Relative difference between Fusion and Raw data (spectral line removed)', fontsize=16)
ax_signal_2.legend(fontsize=9)
ax_signal_2.grid(True)
filtered_relative_difference = (continium_fusion_cube[nonlines_idx] - continuum_cube_interp) /continuum_cube_interp
print(f"Mean filtered_ relative difference = {np.mean(filtered_relative_difference)}")
ax_signal2_2 = fig2.add_subplot(gs2[1,1])  # occupe les deux lignes de la colonne de droite
ax_signal2_2.plot(wavelength[nonlines_idx], filtered_relative_difference, label='Relative Difference', color='green', drawstyle='steps-mid')
ax_signal2_2.set_xlabel('Wavelength (µm)', fontsize=14)
ax_signal2_2.set_ylabel('Flux', fontsize=14)
ax_signal2_2.set_title('Relative difference between Fusion and Filtered Raw data (spectral line removed)', fontsize=16)
ax_signal2_2.legend(fontsize=9)
ax_signal2_2.grid(True)


plt.tight_layout()




import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def find_line_bounds(wavelength, flux, center_index, threshold_ratio=0.05):
    """
    Trouve les bornes gauche/droite d'une raie centrée à center_index
    en s'arrêtant quand le signal retombe à threshold_ratio * amplitude.
    """
    peak_flux = flux[center_index]
    amplitude = np.abs(peak_flux)
    threshold = amplitude * threshold_ratio

    # Chercher les bornes gauche et droite
    i_left = center_index
    while i_left > 0 and np.abs(flux[i_left]) > threshold:
        i_left -= 1

    i_right = center_index
    while i_right < len(flux) - 1 and np.abs(flux[i_right]) > threshold:
        i_right += 1

    return i_left, i_right


def compare_lines_fluxes_ref_on_corrected(
    wl_corr, flux_corr, wl_raw, flux_raw,
    prominence=0.1, threshold_ratio=0.03
):
    """
    Compare les flux des raies détectées sur le signal corrigé.
    La largeur de la raie est déterminée sur le signal brut.
    
    Paramètres :
      wl_corr, flux_corr : spectre corrigé (raies fines)
      wl_raw, flux_raw : spectre brut (raies plus larges)
      prominence : seuil relatif pour la détection des raies
      threshold_ratio : seuil relatif pour trouver les bornes dans le signal brut

    Retour :
      Liste de dictionnaires {center, flux_corr, flux_raw, ratio, ...}
    """
    # 1️⃣ Détection des pics sur le signal corrigé
    peaks, _ = find_peaks(np.abs(flux_corr), prominence=prominence * np.max(np.abs(flux_corr)))
    interp_corr = interp1d(wl_corr, flux_corr, bounds_error=False, fill_value=0.0)

    results = []

    for p in peaks:
        wl_center = wl_corr[p]

        # Trouver le point le plus proche dans le signal brut
        idx_raw = np.argmin(np.abs(wl_raw - wl_center))

        # Trouver les bornes de la raie dans le signal brut
        i_left, i_right = find_line_bounds(wl_raw, flux_raw, idx_raw, threshold_ratio)

        # Définir la zone d’intégration (dans le signal brut)
        wl_min, wl_max = wl_raw[i_left], wl_raw[i_right]

        # Interpolation du flux corrigé sur la même plage
        mask_raw = (wl_raw >= wl_min) & (wl_raw <= wl_max)
        wl_segment = wl_raw[mask_raw]
        flux_corr_interp = interp_corr(wl_segment)
        flux_raw_segment = flux_raw[mask_raw]

        # Intégration (trapézoïde)
        flux_corr_int = np.trapz(flux_corr_interp, wl_segment)
        flux_raw_int = np.trapz(flux_raw_segment, wl_segment)

        results.append({
            "center": wl_center,
            "wl_segment": wl_segment,
            "flux_corr": flux_corr_int,
            "flux_raw": flux_raw_int,
            "ratio": flux_corr_int / flux_raw_int if flux_raw_int != 0 else np.nan
        })

    return results


def compare_spectrale_lines_flux(
        fusion_wl, sline_fusion,
        raw_wl, sline_raw,
        prominence=0.1,
        threshold_ratio=0.5):
    
    fusion_peak_idx = np.where(sline_fusion != 0)[0]

    results = []
    for peak in fusion_peak_idx:
        peak_wl = fusion_wl[peak]
        mask_fusion = sline_fusion==sline_fusion[peak]
        
        raw_peak_idx = np.argmin(np.abs(raw_wl -peak_wl))

        # Find Sum fluw of raw data for the spectral line
        peak_flux = sline_raw[raw_peak_idx]
        threshold = peak_flux*threshold_ratio
        i_left = raw_peak_idx
        while i_left > 0 and np.abs(sline_raw[i_left]) > threshold:
            i_left -= 1

        i_right = raw_peak_idx
        while i_right < len(sline_raw) - 1 and np.abs(sline_raw[i_right]) > threshold:
            i_right += 1

        wl_min, wl_max = raw_wl[i_left], raw_wl[i_right]
        mask_raw = (raw_wl >= wl_min) & (raw_wl <= wl_max)
        raw_slice = slice(i_left, i_right)

        sum_raw_flux = np.sum(sline_raw[mask_raw])
        flux_fusion = sline_fusion[peak]
        
        width = i_right-i_left

        fusion_slice = slice(peak-width, peak+width)
        mask_fusion[peak-i_left:peak+i_right] = 1
        flux_fusion_int = np.trapz(sline_fusion[fusion_slice], fusion_wl[fusion_slice])
        flux_raw_int = np.trapz(sline_raw[raw_slice], raw_wl[raw_slice])
        print(f"Shape mask_fusion = {mask_fusion.shape}")
        print(f"Shape fusion_wl = {fusion_wl.shape}")
        print(f"Shape raw_wl = {raw_wl.shape}")
        print(f"Shape sline_raw[mask_raw] = {sline_raw[raw_slice].shape} : {raw_wl[raw_slice]}")
        print(f"Shape mask_raw = {mask_raw.shape}")
        # plt.figure()
        # plt.plot(fusion_wl[fusion_slice], sline_fusion[fusion_slice])
        # plt.plot(raw_wl[raw_slice], sline_raw[raw_slice])
        # plt.show()

        results.append(
            {
            "center": peak_wl,
            "flux_fusion": flux_fusion,
            "flux_raw": sum_raw_flux,
            "ratio": flux_fusion / sum_raw_flux if sum_raw_flux != 0 else np.nan,
            "width" : width,
            "flux_fusion_int": flux_fusion_int,
            "flux_raw_int": flux_raw_int,
            "ratio_int": flux_fusion_int/flux_raw_int  if flux_raw_int != 0 else np.nan,
        }
        )

    return results


sum_lines = np.zeros((templates.shape[1]))
for i in range(templates.shape[0]): # Car il y a 8 spectral lines dans les templates
    if np.sum(templates[i]!=0) <10:
        sum_lines += templates[i]

lines_idx = np.where(sum_lines != 0)[0]

continium_fusion_cube = mean_fusion_cube.copy()
continium_fusion_cube[lines_idx] = (mean_fusion_cube[lines_idx -1] + mean_fusion_cube[lines_idx +1])/2
spectral_line_fusion = mean_fusion_cube - continium_fusion_cube

print(f"Before compare spectral_line_raw shape is {spectral_line_raw.shape}")
results = compare_spectrale_lines_flux(
    wavelength, spectral_line_fusion,
    wavelength[nonlines_idx], spectral_line_raw,
    prominence=0.1,
    threshold_ratio=0.03
)


# results = compare_lines_fluxes_ref_on_corrected(
#     wavelength[nonlines_idx], spectral_line_fusion,
#     wavelength[nonlines_idx], spectral_line_raw,
#     prominence=0.1,         # ajuste selon ton bruit
#     threshold_ratio=0.05    # 5% de l’amplitude
# )

for r in results:
    print(f"Raie à {r['center']:.4f} µm :")
    print(f"  Flux corrigé = {r['flux_fusion']:.3e}")
    print(f"  Flux brut    = {r['flux_raw']:.3e}")
    print(f"  Ratio corr/brut = {r['ratio']:.3f}")
    print(f"  Flux flux_fusion_int = {r['flux_fusion_int']:.3e}")
    print(f"  Flux flux_raw_int    = {r['flux_raw_int']:.3e}")
    print(f"  ratio_int corr/brut = {r['ratio_int']:.3f}")
    print(f"  Width = {r['width']} slices\n")


plt.figure()
plt.plot(wavelength, spectral_line_fusion)
plt.plot(wavelength[nonlines_idx], spectral_line_raw)

plt.show()
