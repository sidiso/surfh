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



hdul = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Scan/ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits')
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
# wavel = wavel[:9935] # 1A to 4B
# raw_data_cube = data_cube[:9935,:,:]
wavel = wavel
raw_data_cube = data_cube


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
# cube_vizualisation.plot_cube(masked_array, wavel)
# plt.show()

masked_array = interpolate_negatives(masked_array)
masked_array[np.isnan(masked_array)] = 0
masked_array_fitlered_data_cube = ndimage.median_filter(masked_array.copy(), size=15, axes=[0])

masked_array_fitlered_data_cube_SS4 = masked_array_fitlered_data_cube
# masked_array_fitlered_data_cube_SS4 = masked_array_fitlered_data_cube_SS4[:-25,:,:] # remove last 25 slices to match the wavel axis
wavel_SS4 = wavel
# wavel_SS4 = wavel_SS4[:-25] # remove last 25 slices to match the wavel axis


cube_vizualisation.plot_cube(masked_array_fitlered_data_cube_SS4, wavel_SS4)
plt.show()
masked_array_fitlered_data = rearrange(masked_array_fitlered_data_cube_SS4, 'L I J -> (I J) L') # from spectro data
# plt.figure()
# plt.title("Masked data mean spectra")
# plt.plot(wavel, np.nanmean(masked_array, axis=(1,2)))
# plt.figure()
# plt.title("Filtered Masked data mean spectra")
# plt.plot(wavel_SS4, np.nanmean(masked_array_fitlered_data_cube_SS4, axis=(1,2)))
# plt.show()


# Range of components to test
component_range = range(1, 12)  # Adjust based on how many tests you want to run

# List to store the reconstruction errors
reconstruction_errors = []
mre_reconstruction_errors = []

# Compute NMF for different numbers of components and calculate the reconstruction errors
for n_components in component_range:
    nmf = NMF(n_components=n_components, init='random', random_state=42)
    W = nmf.fit_transform(masked_array_fitlered_data)  # W is the weight matrix
    H = nmf.components_          # H is the feature matrix (components)
    reconstructed = W @ H        # Reconstruct the original matrix
    error = np.linalg.norm(masked_array_fitlered_data - reconstructed)  # Frobenius norm
    mre_error = np.mean(np.divide((masked_array_fitlered_data-reconstructed), masked_array_fitlered_data, out=np.zeros_like(masked_array_fitlered_data), where=masked_array_fitlered_data!=0))

    reconstruction_errors.append(error)
    mre_reconstruction_errors.append(mre_error)

# Plotting the reconstruction errors
plt.figure(figsize=(10, 5))
plt.plot(component_range, reconstruction_errors, marker='o')
plt.title('Reconstruction Errors by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.grid(True)

# Plotting the reconstruction errors
plt.figure(figsize=(10, 5))
plt.plot(component_range, mre_reconstruction_errors, marker='o')
plt.title('Mean relative error Reconstruction Errors by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('MRE')
plt.grid(True)
plt.show()


from sklearn.decomposition import NMF

# Initialize NMF with the desired number of components
nmf = NMF(n_components=6, init='random', random_state=0, max_iter=1000)

# Fit NMF model to your data
nmf.fit(masked_array_fitlered_data)

# Extract the components (eigenvectors)
components = nmf.components_

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
np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4ABC_6_templates.npy', components)
np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy', wavel_SS4)
# print(wavel_SS4)
