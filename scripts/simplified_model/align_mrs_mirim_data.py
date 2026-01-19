from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os 

from astropy.io import fits
from scipy.ndimage import rotate
from skimage.transform import rescale

from astropy import units as u

from surfh.Models import wavelength_mrs
from surfh.Vizualisation.cube_vizualisation import plot_cube
from surfh.ToolsDir.alignment import interactive_align




""" Chargement des données fits."""
mrs_fits_file = Path(
    "/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/"
    "Full_scan_NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits"
)

mirim_fits_file = Path(
    "/home/nmonnier/Data/JWST/NGC_7023/Fusion/Raw_MIRIM/"
    "Level3_F1000W_i2d_aligned.fits"
)

mirim_fits_path = Path("/home/nmonnier/Data/JWST/NGC_7023/Fusion/Raw_MIRIM/")

hdul = fits.open(mrs_fits_file)
mrs_header = hdul[1].header
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
# rotated_cube = rotate(
#     raw_cube[2000],
#     angle=rotation_angle,
#     reshape=False,
#     axes=(1, 0),
#     order=3,
#     mode="nearest"
# )
foi_rotated_cube = rotated_cube[:,12:175, 68:96].copy()
res_pix = np.abs(mrs_header['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
scale_factor = res_pix / 0.1
foi_rotated_cube = rescale(
    foi_rotated_cube, 
    scale=(1, scale_factor, scale_factor), 
    order=3,              # interpolation bicubique
    preserve_range=True,  # garder les valeurs d’intensité
    anti_aliasing=True
)



rotated_cube = np.sum(rotated_cube, axis=0)
# """ Sélection du FoV commun aux 4 canaux """
foi_cube = rotated_cube[12:175, 68:96]



res_pix = np.abs(mrs_header['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
print(res_pix)
scale_factor = res_pix / 0.1
foi_cube = rescale(
    foi_cube, 
    scale=scale_factor, 
    order=3,              # interpolation bicubique
    preserve_range=True,  # garder les valeurs d’intensité
    anti_aliasing=True
)



hdul_mirim = fits.open(mirim_fits_file)
mirim_header = hdul_mirim[1].header
raw_mirim = hdul_mirim[1].data
res_pix = np.abs(mirim_header['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
print(res_pix)
scale_factor = res_pix / 0.1

raw_mirim_rescale = rescale(
    raw_mirim, 
    scale=scale_factor, 
    order=3,              # interpolation bicubique
    preserve_range=True,  # garder les valeurs d’intensité
    anti_aliasing=True
)

raw_mirim_rescale = rotate(
    raw_mirim_rescale,
    angle=171.5+13.5,
    reshape=False,
    axes=(1, 0),
    order=3,
    mode="nearest"
)

raw_mirim_rescale = raw_mirim_rescale[312:312+212, 406:442]

# for miri_file in os.listdir(mirim_fits_path):
#     hdul_mirim = fits.open(mirim_fits_path/ miri_file)
#     mirim_header = hdul_mirim[1].header
#     raw_mirim = hdul_mirim[1].data
#     res_pix = np.abs(mirim_header['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
#     print(res_pix)
#     scale_factor = res_pix / 0.1

#     raw_mirim_rescale = rescale(
#         raw_mirim, 
#         scale=scale_factor, 
#         order=3,              # interpolation bicubique
#         preserve_range=True,  # garder les valeurs d’intensité
#         anti_aliasing=True
#     )

#     raw_mirim_rescale = rotate(
#         raw_mirim_rescale,
#         angle=171.5+13.5,
#         reshape=False,
#         axes=(1, 0),
#         order=3,
#         mode="nearest"
#     )

#     raw_mirim_rescale = raw_mirim_rescale[312:312+212, 406:442]
#     filename = os.path.splitext(miri_file)[0]

#     # np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Corrected_MIRIM/simplified/Rescaled_' + filename + '.npy', raw_mirim_rescale)


# Save MRS data 
np.save('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Rescaled_MRS_data.npy', foi_rotated_cube)

# Pad rotated cube to be the same size as mirim
pad_y = (raw_mirim_rescale.shape[0] - foi_cube.shape[0])
pad_x = (raw_mirim_rescale.shape[1] - foi_cube.shape[1])
pad_width = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2))
padded_foi_cube = np.pad(foi_cube, pad_width, mode='constant', constant_values=0)

print(f"MRS data shape = {foi_cube.shape}, MIRIM data shape = {raw_mirim_rescale.shape}")
interactive_align(padded_foi_cube, raw_mirim_rescale)
