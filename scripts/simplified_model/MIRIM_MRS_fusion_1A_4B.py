import numpy as np 
import os

from surfh.Models.simplifiedMIRIM import Mirim_Model_For_Fusion
from surfh.Models.simplifiedMRS import Spectro_Model_3
from surfh.Algorithm.criteria_simplified import QuadCriterion2
from surfh.Models.lmm import LinearMixingModel
from surfh.Vizualisation.cube_vizualisation import plot_cube
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits
import matplotlib.pyplot as plt

import time
import pathlib

fusion_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'

pce_mirim_dir  = fusion_dir + 'Templates/MIRIM/'
wavel_axis = np.load(fusion_dir + 'Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy')
pce_spectro = np.ones_like(wavel_axis)

psfs = np.load(fusion_dir + 'PSF/psfs_pixscale0.1_npix_212x36_chan_1ABC_2ABC_3ABC_4ABC.npy')[:len(wavel_axis),:,:]

templates = np.load(fusion_dir + 'Templates/NMF/full_scan_simplified_ch1a_to_ch4b_12_nmf_components_and_8_spectral_lines.npy')


list_pce = []
for file in sorted(os.listdir(pce_mirim_dir)) :
    print(f'Load PCE file for from file {file} ')
    list_pce.append(np.load(pce_mirim_dir+file)[0])
pce = np.array(list_pce)
pce_mirim = pce[:-1, :len(wavel_axis)]

imshape = (212,36)

L_SS = 5 # Sous-échantillonnage spectrale pour la simulation

print("Shapes before SS: ")
print("psfs shape = ", psfs.shape)
print("wavel_axis shape = ", wavel_axis.shape)
print("templates shape = ", templates.shape)
print("PCE Shape = ", pce_mirim.shape)


psfs_SS = psfs[::L_SS,:,:]  # sous-échantillonnage des psfs pour la simulation
wavel_axis_SS = wavel_axis[::L_SS]  # sous-échantillonnage des longueurs d'onde pour la simulation
templates_SS = templates[:,::L_SS]  # sous-échantillonnage des templates pour la simulation
pce_spectro_SS = pce_spectro[::L_SS]  # sous-échantillonnage du pce pour la simulation
pce_mirim_SS = pce_mirim[:,::L_SS]  # sous-échantillonnage du pce pour la simulation

print("Shapes after SS: ")
print("psfs_SS shape = ", psfs_SS.shape)
print("wavel_axis_SS shape = ", wavel_axis_SS.shape)
print("templates_SS shape = ", templates_SS.shape)
print("pce_spectro_SS shape = ", pce_spectro_SS.shape)
print("pce_mirim_SS shape = ", pce_mirim_SS.shape)




decim = 2
# facteurs de décimations spatiales pour le modèle spectro
di = decim
dj = decim

# reshape mirim model for the fusion model
start = time.time()
# h_int = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/hint_124pix_13templates_SS10.npy')
h_int = None
mirim_model_for_fusion = Mirim_Model_For_Fusion(
    psfs_SS, pce_mirim_SS, wavel_axis_SS, templates_SS, imshape, di, dj, precomputed_H_int = h_int, low_mem=True
)
# np.save('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/hint_124pix_13templates_SS10.npy', mirim_model_for_fusion.H_int)
end = time.time()
print(f"Time to create mirim model for fusion: {end - start} seconds")


# creation of spectro model from scratch
start = time.time()
spectro_model = Spectro_Model_3(
    psfs_SS, pce_spectro_SS, di, dj, wavel_axis_SS, templates_SS, imshape
)
end = time.time()
print(f"Time to create spectro model: {end - start} seconds")

# Load MIRIM data
start = time.time()
y_mirim_list = []
for mirim_data_file in sorted(os.listdir('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Corrected_MIRIM/simplified')):
    print(f'Load MIRIM data from file {mirim_data_file} ')
    y_mirim_list.append(np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Corrected_MIRIM/simplified/' + mirim_data_file))
y_mirim = np.array(y_mirim_list)
print("y_mirim shape = ", y_mirim.shape)
y_mirim = y_mirim[:-1]
print("y_mirim shape after removing last channel = ", y_mirim.shape)
end = time.time()
print(f"Time to compute y_mirim: {end - start} seconds")


# Load and preprocess MRS data
start = time.time()
raw_y_spectro = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Rescaled_MRS_data.npy')
end = time.time()
print(f"Time to compute y_spectro: {end - start} seconds")
print("y_spectro shape = ", raw_y_spectro.shape)
raw_y_spectro = raw_y_spectro[:len(wavel_axis),:,:]
raw_y_spectro = raw_y_spectro[::L_SS, :, :]

# N, H, W = raw_y_spectro.shape
# H4, W4 = (H//decim)*decim, (W//4)*decim
# tab2 = raw_y_spectro[:, :H4, :W4]

# y_spectro = tab2.reshape(N, H4//4, 4, W4//4, 4).mean(axis=(2, 4))
from skimage.measure import block_reduce
import numpy as np

y_spectro = block_reduce(raw_y_spectro, block_size=(1, decim, decim), func=np.sum)

print("y_spectro shape after decimation: ", y_spectro.shape)

weight_mirim = np.sum(y_mirim)
weight_spectro = np.sum(y_spectro)

mu_imager = 1
mu_spectro = 1 * (weight_mirim / weight_spectro)

list_mu_reg = [1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
list_mu_spectro = [mu_spectro * factor for factor in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]]

mu_spectro = list_mu_spectro[2]

print(f"mu_spectro = {mu_spectro}")
mu_reg = 1e1


start = time.time()
quadcriterion = QuadCriterion2(
    mu_imager,
    y_mirim,
    mirim_model_for_fusion,
    mu_spectro,
    y_spectro,
    spectro_model,
    list_mu_reg[2],
    printing = False,
    gradient = "separated"
)
end = time.time()
print(f"Time to create quadcriterion: {end - start} seconds")

start = time.time()
quadcrit_rec_maps = quadcriterion.run_expsol()
end = time.time()
print(f"Time to run exp sol: {end - start} seconds")
print(quadcrit_rec_maps.shape)
# Linear_Mixture_Model.mapsToCube(quadcrit_rec_maps, templates_SS)

from surfh.ToolsDir.matrix_op import lmm_maps2cube
res_cube = lmm_maps2cube(quadcrit_rec_maps, templates_SS)

# Save results
result_dir = f'MIRIM_MRS_EXP_Simplified_mu_{mu_reg}_muImg_{mu_imager}_muMRS_{mu_spectro:.2f}/'

path = pathlib.Path('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/' + result_dir)
path.mkdir(parents=True, exist_ok=True)
metadata = {"WAVELENGTH": wavel_axis_SS}
save_numpy_to_fits(res_cube, metadata, path/'Reconstructed_cube.fits', masks=None)

# true_cube = LinearMixingModel.mapsToCube(quadcrit_rec_maps, templates_SS)
# plot_cube(true_cube, wavel_axis_SS, title='Reconstructed Cube')

y_mrs_mean_spectrum = np.mean(raw_y_spectro, axis=(1,2))
xy_mirim = mirim_model_for_fusion.forward(quadcrit_rec_maps)

plt.figure()
plt.plot(wavel_axis_SS, np.mean(res_cube, axis=(1,2)), label='Reconstructed Spectrum')
plt.plot(wavel_axis_SS, y_mrs_mean_spectrum, label='Raw MRS Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Mean Spectrum over spatial dimensions')
plt.legend()
plt.figure()
plt.plot(wavel_axis_SS, (np.mean(res_cube, axis=(1,2))- y_mrs_mean_spectrum)/y_mrs_mean_spectrum, label='Error Spectrum')
print("Mean absolute relative error spectrum:", np.mean(np.abs((np.mean(res_cube, axis=(1,2))- y_mrs_mean_spectrum)/y_mrs_mean_spectrum)))
plt.show()

# filter_wavelength = 7.7
# idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
# fig, axes = plt.subplots(3, 3, figsize=(24, 8))
# plt.subplots_adjust(wspace=0.15, hspace=0.25)
# im0 = axes[0, 0].imshow(raw_y_spectro[idx_filter], origin='lower', cmap='viridis')
# axes[0, 0].set_title(f'MRS Raw Data - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im0, ax=axes[0, 0], fraction=0.025, pad=0.02)

# im1 = axes[0, 1].imshow(res_cube[idx_filter], origin='lower', cmap='viridis')
# axes[0, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im1, ax=axes[0, 1], fraction=0.025, pad=0.02)

# im2 = axes[0, 2].imshow(y_mirim[1], origin='lower', cmap='viridis')
# axes[0, 2].set_title(f'MIRIM - Filter 770W')
# plt.colorbar(im2, ax=axes[0, 2], fraction=0.025, pad=0.02)

# filter_wavelength = 12.8
# idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
# im3 = axes[1, 0].imshow(raw_y_spectro[idx_filter], origin='lower', cmap='viridis')
# axes[1, 0].set_title(f'MRS Raw Data - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im3, ax=axes[1, 0], fraction=0.025, pad=0.02)

# im4 = axes[1, 1].imshow(res_cube[idx_filter], origin='lower', cmap='viridis')
# axes[1, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im4, ax=axes[1, 1], fraction=0.025, pad=0.02)
# im5 = axes[1, 2].imshow(y_mirim[4], origin='lower', cmap='viridis')
# axes[1, 2].set_title(f'MIRIM - Filter F1280W')
# plt.colorbar(im5, ax=axes[1, 2], fraction=0.025, pad=0.02)


# filter_wavelength = 21.0
# idx_filter = np.argmin(np.abs(wavel_axis_SS - filter_wavelength))
# im3 = axes[2, 0].imshow(raw_y_spectro[idx_filter], origin='lower', cmap='viridis')
# axes[2, 0].set_title(f'MRS Raw Data - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im3, ax=axes[2, 0], fraction=0.025, pad=0.02)

# im4 = axes[2, 1].imshow(res_cube[idx_filter], origin='lower', cmap='viridis')
# axes[2, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[idx_filter]:.2f}')
# plt.colorbar(im4, ax=axes[2, 1], fraction=0.025, pad=0.02)
# im5 = axes[2, 2].imshow(y_mirim[-1], origin='lower', cmap='viridis')
# axes[2, 2].set_title(f'MIRIM - Filter F21000W')
# plt.colorbar(im5, ax=axes[2, 2], fraction=0.025, pad=0.02)
# # plt.tight_layout()


filters_name = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W']
fig, axes = plt.subplots(3, 8, figsize=(15, 5))
for filter in range(y_mirim.shape[0]):
    im = axes[0, filter].imshow(y_mirim[filter][2:-2, 2:-2], origin='lower', cmap='viridis')
    axes[0, filter].set_title(f'Filter {filters_name[filter]}')
    plt.colorbar(im, ax=axes[0, filter], fraction=0.046, pad=0.04)

    im2 = axes[1, filter].imshow(xy_mirim[filter][2:-2, 2:-2], origin='lower', cmap='viridis')
    axes[1, filter].set_title(f'Reconstructed Filter {filters_name[filter]}')
    plt.colorbar(im2, ax=axes[1, filter], fraction=0.046, pad=0.04)
    im3 = axes[2, filter].imshow((y_mirim[filter][2:-2, 2:-2]- xy_mirim[filter][2:-2, 2:-2])/y_mirim[filter][2:-2, 2:-2], origin='lower', cmap='RdBu_r')
    axes[2, filter].set_title(f'Residuals Filter {filters_name[filter]}')
    plt.colorbar(im3, ax=axes[2, filter], fraction=0.046, pad=0.04)

    print(f'Filter {filters_name[filter]} - Mean absolute relative residual: {np.mean(np.abs((y_mirim[filter][2:-2, 2:-2]- xy_mirim[filter][2:-2, 2:-2])/y_mirim[filter][2:-2, 2:-2]))}')
# plt.tight_layout()


# Coupe vericale du cube à une lingueur d'onde donnée  et comparaison avec les données MRS brutes et le filtre MIRIM correspondant
pixel_cut = 36-12
wavelength_cut = 7.7
idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
cube_cut = res_cube[idx_wavelength_cut, :, :]
plt.figure()
plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 7.7 um', drawstyle='steps-mid')
plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 7.7 um', drawstyle='steps-mid')
plt.plot(y_mirim[1][:, pixel_cut], label='MIRIM Data Filter F770W Cut at 7.7 um', drawstyle='steps-mid')
plt.xlabel('Pixel Y')
plt.ylabel('Intensity')
plt.title('Vertical Cut at X=18')
plt.legend()    

# Coupe à 21.0 um
wavelength_cut = 21.0
idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
cube_cut = res_cube[idx_wavelength_cut, :, :]
plt.figure()
plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 21.0 um', drawstyle='steps-mid')
plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 21.0 um', drawstyle='steps-mid')
plt.plot(y_mirim[-1][:, pixel_cut], label='MIRIM Data Filter F2100W Cut at 21.0 um', drawstyle='steps-mid')
plt.xlabel('Pixel Y')
plt.ylabel('Intensity')
plt.title('Vertical Cut at X=18')
plt.legend()


# Coupe à 18.0 um
wavelength_cut = 18.0
idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
cube_cut = res_cube[idx_wavelength_cut, :, :]
plt.figure()
plt.plot(cube_cut[:, pixel_cut], label='Reconstructed Cube Cut at 18.0 um', drawstyle='steps-mid')
plt.plot(raw_y_spectro[idx_wavelength_cut, :, pixel_cut], label='Raw MRS Data Cut at 18.0 um', drawstyle='steps-mid')
plt.plot(y_mirim[-1][:, pixel_cut], label='MIRIM Data Filter F2100W Cut at 18.0 um', drawstyle='steps-mid')
plt.xlabel('Pixel Y')
plt.ylabel('Intensity')
plt.title('Vertical Cut at X=18')
plt.legend()



# Coupe à 15.0 um
wavelength_cut = 15.0
idx_wavelength_cut = np.argmin(np.abs(wavel_axis_SS - wavelength_cut))
cube_cut = res_cube[idx_wavelength_cut, :, :]
plt.figure()
plt.plot(np.flip(cube_cut[:, pixel_cut]), label='Reconstructed Cube Cut at 15.0 um', drawstyle='steps-mid')
plt.plot(np.flip(raw_y_spectro[idx_wavelength_cut, :, pixel_cut]), label='Raw MRS Data Cut at 15.0 um', drawstyle='steps-mid')
# plt.plot(y_mirim[-1][:, pixel_cut], label='MIRIM Data Filter F2100W Cut at 15.0 um', drawstyle='steps-mid')
plt.xlabel('Pixel Y')
plt.ylabel('Intensity')
plt.title('Vertical Cut at X=18')
plt.legend()



# Show cut in the images
plt.figure()
plt.imshow(np.fliplr(cube_cut), origin='lower', cmap='viridis')
plt.axvline(x=12, color='r', linestyle='--', label='Cut Position X=18')
plt.title(f'Reconstructed Cube at {wavel_axis_SS[idx_wavelength_cut]:.2f} um')
plt.colorbar(label='Intensity') 

plot_cube(res_cube, wavel_axis_SS, title='Reconstructed Cube', show=False)
plot_cube(raw_y_spectro, wavel_axis_SS, title='Raw MRS Cube', show=False)


plt.show()
