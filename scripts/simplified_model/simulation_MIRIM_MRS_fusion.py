import numpy as np 
import os

from surfh.Models.simplifiedMIRIM import Mirim_Model_For_Fusion
from surfh.Models.simplifiedMRS import Spectro_Model_3
from surfh.Algorithm.criteria_simplified import QuadCriterion2
from surfh.Models.lmm import LinearMixingModel
from surfh.Vizualisation.cube_vizualisation import plot_cube
import matplotlib.pyplot as plt

import time

fusion_dir = '/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/'

pce_mirim_dir  = fusion_dir + 'Templates/MIRIM/'
wavel_axis = np.load(fusion_dir + 'Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy')
pce_spectro = np.ones_like(wavel_axis)
psfs = np.load(fusion_dir + 'PSF/psfs_pixscale0.1_npix_124_chan_1ABC_2ABC_3ABC_4ABC.npy')[:len(wavel_axis),:,:]

templates = np.load(fusion_dir + 'Templates/simulation_templates.npy')

true_maps = np.load(fusion_dir + 'Templates/simulation_maps.npy')
true_maps = true_maps[:, :psfs.shape[1], :psfs.shape[2]]
print("True maps shape = ", true_maps.shape)

list_pce = []
for file in sorted(os.listdir(pce_mirim_dir)) :
    print(f'Load PCE file for from file {file} ')
    list_pce.append(np.load(pce_mirim_dir+file)[0])
pce = np.array(list_pce)
pce_mirim = pce[:-1, :len(wavel_axis)]

print("PCE Shape = ", pce.shape)
imshape = (124,124)

L_SS = 10 # Sous-échantillonnage spectrale pour la simulation

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

for i in range(templates.shape[0]):
    if i > 4:
        idx_sl = np.argmin(np.abs(templates[i,:] - np.max(templates[i,:])))
        idx_sl_SS = idx_sl // L_SS
        templates_SS[i, idx_sl_SS] = templates[i, idx_sl]



decim = 4
# facteurs de décimations spatiales pour le modèle spectro
di = decim
dj = decim

# reshape mirim model for the fusion model
start = time.time()
h_int = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/hint_124pix_13templates_SS10.npy')
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

start = time.time()
y_mirim = mirim_model_for_fusion.forward(true_maps)
end = time.time()
print(f"Time to compute y_mirim: {end - start} seconds")

start = time.time()
y_spectro = spectro_model.forward(true_maps)
end = time.time()
print(f"Time to compute y_spectro: {end - start} seconds")
print("y_spectro shape = ", y_spectro.shape)

weight_mirim = np.sum(y_mirim)
weight_spectro = np.sum(y_spectro)

mu_imager = 1
mu_spectro = 1 * (weight_mirim / weight_spectro)
print(f"mu_spectro = {mu_spectro}")
mu_reg = 1


start = time.time()
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
end = time.time()
print(f"Time to create quadcriterion: {end - start} seconds")

start = time.time()
quadcrit_rec_maps = quadcriterion.run_expsol()
end = time.time()
print(f"Time to run exp sol: {end - start} seconds")
print(quadcrit_rec_maps.shape)
# Linear_Mixture_Model.mapsToCube(quadcrit_rec_maps, templates_SS)

res_cube = LinearMixingModel.mapsToCube(quadcrit_rec_maps, templates_SS)
true_cube = LinearMixingModel.mapsToCube(true_maps, templates_SS)
plot_cube(res_cube, wavel_axis_SS, title='Reconstructed Cube')
plot_cube(true_cube, wavel_axis_SS, title='True Cube')
plt.figure()
plt.plot(wavel_axis_SS, np.mean(res_cube, axis=(1,2)), label='Reconstructed Spectrum')
plt.plot(wavel_axis_SS, np.mean(true_cube, axis=(1,2)), label='True Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Mean Spectrum over spatial dimensions')
plt.legend()
plt.figure()
plt.plot(wavel_axis_SS, np.mean(res_cube - true_cube, axis=(1,2)), label='Error Spectrum')
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 5))
im0 = axes[0, 0].imshow(true_cube[100], origin='lower', cmap='viridis')
axes[0, 0].set_title(f'True Data - Lamba =  {wavel_axis_SS[100]}')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(res_cube[100], origin='lower', cmap='viridis')
axes[0, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[100]}')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[0, 2].imshow(res_cube[100] -true_cube[100], origin='lower', cmap='RdBu_r')
axes[0, 2].set_title(f'Residuals - Lamba =  {wavel_axis_SS[100]}')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

im3 = axes[1, 0].imshow(true_cube[900], origin='lower', cmap='viridis')
axes[1, 0].set_title(f'True Data - Lamba =  {wavel_axis_SS[900]}')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

im4 = axes[1, 1].imshow(res_cube[900], origin='lower', cmap='viridis')
axes[1, 1].set_title(f'Res Cube - Lamba =  {wavel_axis_SS[900]}')
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

im5 = axes[1, 2].imshow(res_cube[900] -true_cube[900], origin='lower', cmap='RdBu_r')
axes[1, 2].set_title(f'Residuals - Lamba =  {wavel_axis_SS[900]}')
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
