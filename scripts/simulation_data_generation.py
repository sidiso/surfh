import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

from surfh.Models.lmm import LinearMixingModel
from surfh.Vizualisation import cube_vizualisation

"""
Génération de données de simulation pour la fusion MRS.
On utilise les 4 cartes d'abondances de la bar d'Orion simmulée. 
On utilise les longueurs d'onde qui sont utilisées avec les vraies données MRS auxquelles on été ajouté les longueurs d'ondes des raies spectrales que l'on ajoute.
Les Cartes d'abondances des raies spectrales sont construites à partir de cartes d'abondances existantes avec l'ajout de structures ponctuelles ou étendues.
Pour ne pas alourdir la simulation on utilisera 5 ou 6 raies spectrales avec des cartes d'abondances identiques
"""


Nx = Ny = 125 # Taille en pixels des images simulées. Cette taille est choisie pour être identique à celle des images utilisées avec les vraies données MRS.
step = 0.1  # Pas spatial en arcsec/pixel des images simulées

def orion():
    """Rerturn maps, templates, spatial step and wavelength"""
    path_cube_orion='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
    maps = fits.open(path_cube_orion + "abundances_orion.fits")[0].data

    print(f"There is {maps.shape[0]} maps available of size {maps.shape[1]}x{maps.shape[2]}.")

    h2_map = maps[0]
    if_map = maps[1]
    df_map = maps[2]
    mc_map = maps[3]

    spectrums = fits.open(path_cube_orion + "spectra_mir_orion.fits")[1].data
    wavel_axis = spectrums.wavelength
    print(f"Wavelength axis shape: {wavel_axis.shape}")

    h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
    if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
    df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
    mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

    return (
        np.asarray((h2_map, if_map, df_map, mc_map)),
        np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
        0.025,
        wavel_axis,
    )


maps, tpl, step, wavel_axis = orion()

SS_factor = maps.shape[1] // Nx  # Spatial subsampling factor
maps = maps[:, ::SS_factor, ::SS_factor]  # Downsample maps to desired


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Abundance Maps of Orion Bar Simulation", fontsize=16)
ax[0, 0].imshow(maps[0], cmap='inferno')
ax[0, 0].set_title("H2 Abundance Map")
ax[0, 1].imshow(maps[1], cmap='inferno')
ax[0, 1].set_title("IF Abundance Map")
ax[1, 0].imshow(maps[2], cmap='inferno')
ax[1, 0].set_title("DF Abundance Map")
ax[1, 1].imshow(maps[3], cmap='inferno')
ax[1, 1].set_title("MC Abundance Map")
plt.tight_layout()

# plt.figure(figsize=(8, 5))
# for i, spectrum in enumerate(tpl):
#     plt.plot(wavel_axis, spectrum, label=['H2', 'IF', 'DF', 'MC'][i])
# plt.title("Spectral Templates")
# plt.xlabel("Wavelength (µm)")
# plt.ylabel("Intensity")
# plt.legend()

# plt.show()

cube = LinearMixingModel.mapsToCube(maps, tpl)
# cube_vizualisation.plot_cube(cube, wavel_axis)

sim_templates = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/nmf_NGC7023_1ABC_2ABC_3ABC_4AB_16_templates.npy')
sim_wavelength = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy')

# Select all templates except the 5 and 6 th on all 16
new_templates = sim_templates[[0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15]]
# Make maps for this new templates where from 5 to the last is the maps number 2 repeated
new_maps = np.zeros((new_templates.shape[0], Nx, Ny))
new_maps[0:4, :, :] = maps  # First four are the original maps
for i in range(4, new_templates.shape[0]):
    new_maps[i, :, :] = maps[1, :, :]  # Repeat the DF map  

new_maps *= 100
cube = LinearMixingModel.mapsToCube(new_maps, new_templates)
# cube_vizualisation.plot_cube(cube, wavel_axis)

# save cube 
np.save('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy', cube)
np.save('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_templates.npy', new_templates)
np.save('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_maps.npy', new_maps)