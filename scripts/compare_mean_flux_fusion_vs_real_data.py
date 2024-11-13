import numpy as np
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs
import matplotlib.pyplot as plt
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from astropy.io import fits
from scipy import ndimage



sim_dir_path='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
_, _, wavel_axis, _, _, _ = simulation_data.get_simulation_data(4, 0, sim_dir_path)

# Get indexes of the cube_wavelength for specific wavelength window
indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('1c')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('1c')[-1]))[0]
window_slice = slice(indexes[0]-1, indexes[-1] +1, None) # 
wavel_axis = wavel_axis[window_slice]

fusion_cube = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/lcg_MC_1_MO_1_nit_32_mu_500000000.0/res_cube.npy')
mask = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Masks/binary_mask_1C.npy')
masked_fusion_cube = fusion_cube*mask[np.newaxis,...]
flipped_cube = np.array([np.fliplr(masked_fusion_cube[i]) for i in range(masked_fusion_cube.shape[0])])

abundance_maps = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/lcg_MC_1_MO_1_nit_32_mu_500000000.0/res_x.npy')
mask = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Masks/binary_mask_1C.npy')
masked_fusion_cube = abundance_maps*mask[np.newaxis,...]
flipped_maps = np.array([np.fliplr(masked_fusion_cube[i]) for i in range(masked_fusion_cube.shape[0])])

from matplotlib import colors
fig, axs = plt.subplots(1,2)
fig.suptitle("Cartes d'abondance")
norm = colors.Normalize(vmin=np.min(flipped_maps[0]), vmax=np.max(flipped_maps[0]))
images = []
for idx, ax in enumerate(axs):
    images.append(ax.imshow(np.rot90(flipped_maps[idx], -1), norm=norm))
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=0.1)

plt.figure()
plt.imshow(np.rot90(flipped_maps[0], -1))
plt.colorbar()
plt.figure()
plt.imshow(np.rot90(flipped_maps[1], -1))
plt.colorbar()
plt.show()


with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/Filtered_ChannelCube_ch1-long_s3d.fits', mode='update') as fitsfile:
    data_cube = fitsfile[1].data
    filtered_data_cube = ndimage.median_filter(data_cube.copy(), size=15, axes=[0])
    data_cube = fitsfile[1].data = filtered_data_cube

hdul = fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/ChannelCube_ch1-long_s3d.fits')
data_cube = hdul[1].data

hdr = hdul[1].header
wavel = (np.arange(hdr['NAXIS3']) +hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']

# replace Nan with 0
data_cube[np.isnan(data_cube)] = 0
filtered_data_cube = ndimage.median_filter(data_cube.copy(), size=15, axes=[0])


# Compute the mean of non-zero elements along the 'slice' axis
mean_non_zero_fusion = np.array([
                                    np.mean(slice[slice != 0]) if np.count_nonzero(slice) > 0 else 0
                                    for slice in masked_fusion_cube
                                ])

mean_non_zero_real = np.array([
                                    np.mean(slice[slice != 0]) if np.count_nonzero(slice) > 0 else 0
                                    for slice in filtered_data_cube
                                ])
print(mean_non_zero_fusion.shape)
print(mean_non_zero_real.shape)

cube_vizualisation.plot_cube(np.rot90(flipped_cube, -1, axes=(1,2)), wavel)
plt.plot(wavel, mean_non_zero_fusion, label='fusion')
plt.plot(wavel, mean_non_zero_real, label='real')
plt.legend()


from matplotlib.path import Path

points = [(132,257), (60,129), (247,200), (184,72)]

plt.figure()
plt.imshow(masked_fusion_cube[100])
plt.plot(points[0][0], points[0][1], '.')
plt.plot(points[1][0], points[1][1], '.')
plt.plot(points[2][0], points[2][1], '.')
plt.plot(points[3][0], points[3][1], '.')

# Créer le chemin du polygone avec les points du rectangle
polygon_path = Path(points)

# Déterminer les limites en ligne et en colonne
min_row = min(point[0] for point in points)
max_row = max(point[0] for point in points)
min_col = min(point[1] for point in points)
max_col = max(point[1] for point in points)

# Préparer une liste pour les valeurs dans le rectangle
values_in_rectangle = []

# Parcourir la zone limite
for y in range(min_row, max_row + 1):
    for x in range(min_col, max_col + 1):
        # Vérifier si le point est dans le rectangle orienté
        if polygon_path.contains_point((y, x)):
            values_in_rectangle.append(masked_fusion_cube[:,y, x])
print(len(values_in_rectangle))
print(len(values_in_rectangle[0]))
print(np.array(values_in_rectangle).shape)

plt.figure()
plt.plot(wavel, np.mean(np.array(values_in_rectangle), axis=0).T, label='fusion')
plt.plot(wavel, mean_non_zero_real, label='real')
plt.legend()
plt.show()
