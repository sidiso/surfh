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
raw_data_cube = interpolate_negatives(masked_array)



""" Séparation du continuum et des lignes spectrales par filtrage médian 1D le long de l'axe spectral"""
continiuum_data_cube = ndimage.median_filter(raw_data_cube.copy(), size=15, axes=[0])
spectral_line_data_cube = raw_data_cube - continiuum_data_cube


spectral_line_spectrum = np.nanmean(spectral_line_data_cube, axis=(1, 2))



""" Détection précise des pics dans le spectre des lignes spectrales """
from scipy.signal import find_peaks
from astropy.stats import mad_std
noise = mad_std(spectral_line_spectrum)   # estimation du bruit
peaks, properties = find_peaks(spectral_line_spectrum, height=13*noise, distance=5) # seuil = k * bruit, par ex. k=3 ou 5

""" Changement de shape des données pour appliquer la NMF """
masked_array_fitlered_data = rearrange(continiuum_data_cube, 'L I J -> (I J) L') # from spectro data

""" On teste le nombre de composantes pour minimiser l'erreur de reconstruction. A commenter si on sait déjà le nombre de composantes souhaité """
component_range = range(1, 12)  # Adjust based on how many tests you want to run
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
# plt.show()


""" Application de la NMF avec le nombre de composantes choisi """
# Initialize NMF with the desired number of components
nmf = NMF(n_components=6, init='random', random_state=0, max_iter=1000)
nmf.fit(masked_array_fitlered_data) # Fit NMF model to your data
components = nmf.components_ # Extract the components (eigenvectors)

n_components = components.shape[0]  # ici 6

# On ajoute les raies spectrale dans les composantes à sauvegarder
for peak in range(len(peaks)):
    peak_line = np.zeros(components.shape[1])
    peak_line[peaks[peak]] = spectral_line_spectrum[peaks[peak]]
    components = np.vstack([components, peak_line])


continiuum_data_cube[continiuum_data_cube==0] = np.nan
raw_data_cube[raw_data_cube==0] = np.nan


