import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits



def apply_mask(datacube, mask, wavelengths):
    """Apply a mask to the data cube.

    Parameters
    ----------
    mask : np.ndarray
        A boolean array with the same spatial dimensions as the data cube.
        True values indicate valid data points, while False values indicate masked points.
    """
    if datacube is None:
        raise ValueError("Data cube is not loaded.")
    
    if mask is None:
        return
    
    # if mask .shape != datacube.data.shape[1:]:
    #     raise ValueError(f"Mask shape does not match spatial dimensions of the data cube {mask[:1].shape} vs {datacube.data.shape[1:]}." )
    
    ch_limit = [5.8, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 25]
    # Apply the mask to each wavelength slice
    w_start = w_stop = 0
    for i in range(mask.shape[0]):
        w_stop = np.where(wavelengths < ch_limit[i])[0][-1] +1
        slice_mask = slice(w_start, w_stop)
        datacube[slice_mask,:,:] = datacube[slice_mask,:,:]*mask[i]
        w_start = w_stop
    return datacube

file = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_11_MO_4_Temp_18_nit_200_mu_5.00e+06_SD_True/res_cube.fits'

with fits.open(file) as hdul:
    data = None
    if hdul[0].data is not None:
        data = hdul[0].data
    elif len(hdul) > 1 and hdul[1].data is not None:
        data = hdul[1].data

    if 'MASKS' in hdul:
        mask = hdul['MASKS'].data
        print(f"Load mask of shape {mask.shape} from FITS file.")
    else:
        print("There is no mask in the FITS file.")

    wavelength = None
    if 'WCS-TABLE' in hdul:
        wavel = hdul['WCS-TABLE'].data
        if 'wavelength' in wavel.names:
            wavelength = wavel['wavelength'][0].squeeze() # Because shape can be (N,1) or (1,N)
        
    if 'WAVELENGTH' in hdul:   
        wavelength = hdul['WAVELENGTH'].data


# Print the shapes of the loaded data
print(f"Data shape: {data.shape if data is not None else 'None'}")
print(f"Mask shape: {mask.shape if mask is not None else 'None'}")

# mask = np.ones_like(mask)

from scipy.ndimage import binary_fill_holes
mask[0] = binary_fill_holes(mask[0]).astype(mask.dtype)

data_masked = apply_mask(data, mask, wavelength)
data_masked[data_masked == 0] = np.nan  # Set masked values to NaN for better visualization

n_lambda = data_masked.shape[0]
indices = np.linspace(0, n_lambda-1, 4, dtype=int)

# Création de la figure
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, idx in zip(axes.ravel(), indices):
    image = data_masked[idx, 20:-20, 10:-10]
    
    im = ax.imshow(image, origin='lower', cmap='viridis')
    
    # Titre avec longueur d’onde
    if wavelength is not None:
        wl = wavelength[idx]
        ax.set_title(f"$\\lambda = {wl:.2f}\\ \\mu m$")
    else:
        ax.set_title(f"Index {idx}")
    
    # Colorbar individuelle
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    # Option publication : enlever axes
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# Sauvegarde en haute qualité (important pour A&A)
plt.savefig("/home/nmonnier/Presentations/Fusion_figures/MRS_full_model_fusion.png", dpi=300, bbox_inches='tight')
plt.show()