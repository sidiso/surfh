import numpy as np
from astropy.io import fits
from loguru import logger
from pathlib import Path

from surfh.Models import instru

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import ndimage

from sklearn.decomposition import NMF
from einops import rearrange
from surfh.Models import wavelength_mrs
from surfh.Simulation import simulation_data
from surfh.Vizualisation import cube_vizualisation
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon


import numpy as np
from scipy.ndimage import generic_filter

# Exemple : cube.shape = (bands, height, width)
def interpolate_negatives(cube):
    cube = cube.copy()  # Pour ne pas modifier l'original

    def interpolate_func(values):
        center = values[len(values) // 2]
        if center >= 0:
            return center
        else:
            # Ne garder que les valeurs positives autour
            positives = values[values >= 0]
            return np.mean(positives) if len(positives) > 0 else 0.0

    # Appliquer un filtre 3D (voisinage 3x3x3)
    result = generic_filter(cube, interpolate_func, size=3, mode='mirror')
    return result



hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Scan/NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits')
data_cube = hdul[1].data
cst_data = hdul[1].data
hdr = hdul[1].header

filter_file = False
if filter_file:
    hdul_new = fits.HDUList([hdu.copy() for hdu in hdul])

    # Sélection de l'extension science (souvent nommée 'SCI' ou index 1)
    sci_hdu = hdul_new['SCI'] if 'SCI' in hdul_new else hdul_new[1]

    # Données originales
    data_cube = sci_hdu.data

    # Appliquer le filtre médian le long de l'axe spectral (axe 0)
    filtered_data_cube = ndimage.median_filter(data_cube, size=(15,1,1))

    # Remplacer les données par la version filtrée
    sci_hdu.data = filtered_data_cube

    # Écriture du nouveau fichier
    hdul_new.writeto('/home/nmonnier/Data/JWST/NGC_7023/Scan/ChannelCube_ch1-2-3-4-shortmediumlong_s3d_filtered.fits', overwrite=True)

wavel = np.array(hdul[5].data[0])[0,:,0]

# wavel = wavel[:-10]
# raw_data_cube = data_cube[:-10,:,:] 
# wavel = wavel[2249:8688] # 1C to 3C
# raw_data_cube = data_cube[2249:8688,:,:]
# wavel = wavel[2249:9935] # 1C to 4B
# raw_data_cube = data_cube[2249:9935,:,:]
wavel = wavel[:9935] # 1A to 4B
raw_data_cube = data_cube[:9935,:,:]
# wavel = wavel
# raw_data_cube = data_cube


# plt.figure()
# plt.title("Raw data  cube, slice 100")
# plt.imshow(raw_data_cube[100])


# cube_vizualisation.plot_cube(raw_data_cube, wavel)
# plt.show()

coords = [(36,29), (74,29), (36,75), (74,75)]

from scipy.ndimage import rotate

angle = -37  # angle en degrés

# Créer un nouveau cube pour stocker les résultats
rotated_cube = np.empty_like(raw_data_cube)
raw_data_cube[np.isnan(raw_data_cube)] = 0
# Appliquer la rotation pour chaque tranche spectrale
for i in range(raw_data_cube.shape[0]):
    # Appliquer la rotation en 2D en gardant le centre de la tranche comme centre de rotation
    rotated_cube[i] = rotate(raw_data_cube[i], angle=angle, reshape=False, axes=(1, 0), order=3, mode='nearest')



masked_array = rotated_cube[:, coords[0][1]:coords[2][1], coords[0][0]:coords[3][0]]
spectrum_masked_array = np.nanmean(masked_array, axis=(1,2))
# cube_vizualisation.plot_cube(masked_array, wavel)
# plt.show()

masked_array = interpolate_negatives(masked_array)
masked_array[np.isnan(masked_array)] = 0
masked_array_fitlered_data_cube = ndimage.median_filter(masked_array.copy(), size=15, axes=[0])

masked_array_fitlered_data_cube_SS4 = masked_array_fitlered_data_cube

spectral_line = masked_array - masked_array_fitlered_data_cube

from astropy.stats import mad_std
spectrum = np.nanmean(spectral_line, axis=(1, 2))



# coords2 = [(36,29), (74,29), (36,75), (74,75)]
coords2 = [(62, 17.5), (84, 46), (48, 72), (26.8, 43.4)]


# # --- Paramètres de style (publication ready) ---
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "mathtext.fontset": "cm",
#     "axes.linewidth": 1,
#     "xtick.direction": "in",
#     "ytick.direction": "in",
#     "xtick.top": True,
#     "ytick.right": True,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
# })

# last_slice = raw_data_cube[-1, :, :]

# # --- Définition de la zone d'intérêt ---
# x_min, y_min = coords2[0]  # coin haut-gauche
# x_max, y_max = coords2[-1]  # coin bas-droit
# width = x_max - x_min
# height = y_max - y_min

# # --- Calcul du spectre moyen ---
# # roi = raw_data_cube[:, y_min:y_max, x_min:x_max]
# # spectrum = roi.mean(axis=(1, 2))

# print("len(wavel) =", len(wavel))
# print("len(spectrum) =", len(spectrum))


# # --- Figure ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# # 1. Image du dernier slice
# im = ax1.imshow(last_slice, cmap="viridis", origin="lower")
# # rect = Rectangle((x_min, y_min), width, height,
# #                  linewidth=2, edgecolor='red', facecolor='none',
# #                  label='Channel 1A FoV')
# # ax1.add_patch(rect)
# poly = Polygon(coords2, closed=True, 
#                edgecolor="red", facecolor="none", linewidth=2,
#                label='Channel 1A FoV')
# ax1.add_patch(poly)
# ax1.set_title(r"NGC7023 MRS FoV, $\lambda = 24.48$ (um)", fontsize=16)
# ax1.set_xlabel("x (pixels)", fontsize=14)
# ax1.set_ylabel("y (pixels)", fontsize=14)
# ax1.legend(loc="upper right", fontsize=9, frameon=True)
# cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=9)

# # 2. Spectre moyen
# print(wavel)
# ax2.plot(wavel, spectrum_masked_array, color="black", lw=1.2)
# ax2.set_title("Mean spectrum", fontsize=16)
# ax2.set_xlabel(r"$\lambda$ (um)", fontsize=14)
# ax2.set_ylabel("Intensity (a.u.)", fontsize=14)
# # Ajustements
# plt.tight_layout()
# plt.savefig("/home/nmonnier/Presentations/20250916_INCLASS/NGC7023_MRS_FoV_and_spectrum.png", dpi=300)
# plt.show()

# raise SystemExit


noise = mad_std(spectrum)   # estimation du bruit
from scipy.signal import find_peaks

# seuil = k * bruit, par ex. k=3 ou 5
peaks, properties = find_peaks(spectrum, height=13*noise, distance=5)

# masked_array_fitlered_data_cube_SS4 = masked_array_fitlered_data_cube_SS4[:-25,:,:] # remove last 25 slices to match the wavel axis
wavel_SS4 = wavel
# wavel_SS4 = wavel_SS4[:-25] # remove last 25 slices to match the wavel axis


# cube_vizualisation.plot_cube(masked_array_fitlered_data_cube_SS4, wavel_SS4)
# plt.show()
masked_array_fitlered_data = rearrange(masked_array_fitlered_data_cube_SS4, 'L I J -> (I J) L') # from spectro data

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["DejaVu Serif"],
#     "mathtext.fontset": "cm",
#     "axes.linewidth": 1,
#     "xtick.direction": "in",
#     "ytick.direction": "in",
#     "xtick.top": True,
#     "ytick.right": True,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10
# })

# # --- Plot du spectre avec les pics ---
# plt.figure(figsize=(6,4))
# plt.plot(wavel, spectrum, color="black", lw=1.5, label="Mean spectrum")
# plt.plot(wavel[peaks], spectrum[peaks], "rx", label="Spectral lines")
# plt.xlabel(r"Wavelength $\lambda$ (um)", fontsize=14)
# plt.ylabel("Intensity (a.u.)", fontsize=14)
# plt.title("Mean spectrum without continium", fontsize=16)
# plt.legend(fontsize=9)
# plt.tight_layout()
# plt.savefig("/home/nmonnier/Presentations/20250916_INCLASS/NGC7023_mean_sectrum_without_continium.png", dpi=300)

# # --- Plot du spectre moyen du masked_array ---
# plt.figure(figsize=(6,4))
# plt.plot(wavel, np.nanmean(masked_array, axis=(1,2)), color="black", lw=1.5)
# plt.xlabel(r"Wavelength $\lambda$ (um)", fontsize=14)
# plt.ylabel("Intensity (a.u.)", fontsize=14)
# plt.title("Mean spectrum", fontsize=16)
# plt.tight_layout()
# plt.savefig("/home/nmonnier/Presentations/20250916_INCLASS/NGC7023_mean_sectrum.png", dpi=300)

# # --- Plot du spectre moyen du masked_array filtré ---
# plt.figure(figsize=(6,4))
# plt.plot(wavel_SS4, np.nanmean(masked_array_fitlered_data_cube_SS4, axis=(1,2)), color="black", lw=1.5)
# plt.xlabel(r"Wavelength $\lambda$ (nm)", fontsize=14)
# plt.ylabel("Intensity (a.u.)", fontsize=14)
# plt.title("Median filtered Mean spectrum ", fontsize=16)
# plt.tight_layout()
# plt.savefig("/home/nmonnier/Presentations/20250916_INCLASS/NGC7023_median_filtered_mean_sectrum.png", dpi=300)
# plt.show()


# Range of components to test
component_range = range(1, 12)  # Adjust based on how many tests you want to run

# List to store the reconstruction errors
reconstruction_errors = []
mre_reconstruction_errors = []

# Compute NMF for different numbers of components and calculate the reconstruction errors
# for n_components in component_range:
#     nmf = NMF(n_components=n_components, init='random', random_state=42)
#     W = nmf.fit_transform(masked_array_fitlered_data)  # W is the weight matrix
#     H = nmf.components_          # H is the feature matrix (components)
#     reconstructed = W @ H        # Reconstruct the original matrix
#     error = np.linalg.norm(masked_array_fitlered_data - reconstructed)  # Frobenius norm
#     mre_error = np.mean(np.divide((masked_array_fitlered_data-reconstructed), masked_array_fitlered_data, out=np.zeros_like(masked_array_fitlered_data), where=masked_array_fitlered_data!=0))

#     reconstruction_errors.append(error)
#     mre_reconstruction_errors.append(mre_error)

# # Plotting the reconstruction errors
# plt.figure(figsize=(10, 5))
# plt.plot(component_range, reconstruction_errors, marker='o')
# plt.title('Reconstruction Errors by Number of Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Reconstruction Error')
# plt.grid(True)

# # Plotting the reconstruction errors
# plt.figure(figsize=(10, 5))
# plt.plot(component_range, mre_reconstruction_errors, marker='o')
# plt.title('Mean relative error Reconstruction Errors by Number of Components')
# plt.xlabel('Number of Components')
# plt.ylabel('MRE')
# plt.grid(True)
# plt.show()


from sklearn.decomposition import NMF

# Initialize NMF with the desired number of components
nmf = NMF(n_components=6, init='random', random_state=0, max_iter=1000)

# Fit NMF model to your data
nmf.fit(masked_array_fitlered_data)

# Extract the components (eigenvectors)
components = nmf.components_

n_components = components.shape[0]  # ici 6
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["DejaVu Serif"],
#     "mathtext.fontset": "cm",
#     "axes.linewidth": 1,
#     "xtick.direction": "in",
#     "ytick.direction": "in",
#     "xtick.top": True,
#     "ytick.right": True,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10
# })

# # --- Plot des composantes NMF ---
# fig, axes = plt.subplots(n_components, 1, figsize=(6, 2*n_components), sharex=True)

# for i in range(n_components):
#     ax = axes[i]
#     ax.plot(wavel_SS4, components[i], color='black', lw=1.5)
#     ax.set_ylabel("Intensity (a.u.)", fontsize=10)
#     ax.set_title(f"NMF Components {i+1}", fontsize=14)
#     ax.grid(False)

# axes[-1].set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=10)
# plt.tight_layout()
# plt.savefig("/home/nmonnier/Presentations/20250916_INCLASS/NGC7023_NMF_components.png", dpi=300)
# plt.show()


for peak in range(len(peaks)):
    peak_line = np.zeros(components.shape[1])
    peak_line[peaks[peak]] = spectrum[peaks[peak]]
    components = np.vstack([components, peak_line])

print(f"Components shape after adding spectral lines: {components.shape}")
# plt.figure()
# for i in range(components.shape[0]):
#     plt.plot(wavel_SS4, components[i], label=i)
# plt.legend()
 
# plt.figure()
# plt.plot(wavel_SS4, np.mean(components, axis=0))
# plt.show()
masked_array_fitlered_data_cube[masked_array_fitlered_data_cube==0] = np.nan
masked_array[masked_array==0] = np.nan

scale_f = np.max(np.mean(masked_array_fitlered_data_cube, axis=(1,2)))/np.max(np.mean(components, axis=0))
#scale_f = np.mean(fitlered_data_cube, axis=(1,2))[-1]/np.mean(components, axis=0)[-1]


# SS_wavel = wavel[::4]
# wavel_SS4 = np.load('/home/nmonnier/Data/JWST/Point_source/Fusion/Templates/wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_SS4.npy')
# plt.figure()

# SS_components = np.zeros((components.shape[0], len(SS_wavel)))
# for i in range(components.shape[0]):
#     SS_components[i] = components[i,::4]
# plt.show()
print(f"Components shape: {components.shape}")
print(f"Wavel shape: {wavel_SS4.shape}")
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1C_2ABC_3ABC_6_templates_SS4.npy', components)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1C_2ABC_3ABC_SS4.npy', wavel_SS4)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1C_2ABC_3ABC_4AB_6_templates_SS4.npy', components)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1C_2ABC_3ABC_4AB_SS4.npy', wavel_SS4)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_6_templates_SS4.npy', components)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB_SS4.npy', wavel_SS4)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_14_templates.npy', components)
# np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy', wavel_SS4)
# print(wavel_SS4)
