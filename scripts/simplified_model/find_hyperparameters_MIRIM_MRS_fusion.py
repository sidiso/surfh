import numpy as np 
import os

from surfh.Models.simplifiedMIRIM import Mirim_Model_For_Fusion
from surfh.Models.simplifiedMRS import Spectro_Model_3
from surfh.Algorithm.criteria_simplified import QuadCriterion2
from surfh.Models.lmm import LinearMixingModel
from surfh.Vizualisation.cube_vizualisation import plot_cube
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits
from surfh.ToolsDir.matrix_op import lmm_maps2cube
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




decim = 4
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
print(f"mu_spectro = {mu_spectro}")
mu_reg = 1e1

y_mrs_mean_spectrum = np.mean(raw_y_spectro, axis=(1,2))
list_mu_reg = [1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
list_mu_spectro = [mu_spectro * factor for factor in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]]
map_res_spectro = np.zeros((len(list_mu_reg), len(list_mu_spectro)))
map_res_imager = np.zeros((len(list_mu_reg), len(list_mu_spectro)))
for i, mu_reg in enumerate(list_mu_reg):
    for j, mu_spectro in enumerate(list_mu_spectro):
        print(f'Running fusion with mu_reg = {mu_reg} and mu_spectro = {mu_spectro}')
        
        quadcriterion = QuadCriterion2(
            mu_imager,
            y_mirim,
            mirim_model_for_fusion,
            mu_spectro,
            y_spectro,
            spectro_model,
            mu_reg,
            printing = False,
            gradient = "separated"
        )

        start = time.time()
        quadcrit_rec_maps = quadcriterion.run_expsol()
        end = time.time()
        print(f"Time to run exp sol: {end - start} seconds")

        res_cube = lmm_maps2cube(quadcrit_rec_maps, templates_SS)
        xy_mirim = mirim_model_for_fusion.forward(quadcrit_rec_maps)

        # Save res spectro
        res_spectro = np.mean(np.abs((np.mean(res_cube, axis=(1,2))- y_mrs_mean_spectrum)/y_mrs_mean_spectrum))

        # Save mean values for mirim without borders of 2 pixels
        res_imager = np.mean(np.abs((y_mirim[:,2:-2,2:-2]- xy_mirim[:,2:-2,2:-2])/y_mirim[:,2:-2,2:-2]))

        map_res_spectro[i,j] = res_spectro
        map_res_imager[i,j] = res_imager


plt.figure()
plt.imshow(map_res_spectro, origin='lower', cmap='viridis', extent=(min(list_mu_spectro), max(list_mu_spectro), min(list_mu_reg), max(list_mu_reg)), aspect='auto')
plt.colorbar(label='Mean absolute relative error spectrum')
plt.xlabel('mu_spectro')
plt.ylabel('mu_reg')
plt.title('Mean absolute relative error spectrum vs mu_spectro and mu_reg')
# plt.savefig('Mean_absolute_relative_error_spectrum_vs_mu_spectro_and_mu_reg.png')
plt.figure()
plt.imshow(map_res_imager, origin='lower', cmap='viridis', extent=(min(list_mu_spectro), max(list_mu_spectro), min(list_mu_reg), max(list_mu_reg)), aspect='auto')
plt.colorbar(label='Mean absolute relative error imager')
plt.xlabel('mu_spectro')
plt.ylabel('mu_reg')
plt.title('Mean absolute relative error imager vs mu_spectro and mu_reg')
# plt.savefig('Mean_absolute_relative_error_imager_vs_mu_spectro_and_mu_reg.png')
plt.show()
        





