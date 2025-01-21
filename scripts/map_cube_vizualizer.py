import os
import numpy as np
import matplotlib.pyplot as plt

from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.ToolsDir import utils

res_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/lcg_MC_6_MO_4_Temp_6_nit_250_mu_5.00e+09/'

cube = np.load(res_dir + 'res_cube.npy')
maps = np.load(res_dir + 'res_x.npy')
criterion = np.load(res_dir + 'criterion.npy')




utils.plot_maps(maps)

flipped_cube = np.array([np.fliplr(cube[i]) for i in range(cube.shape[0])])
cube_vizualisation.plot_cube(np.rot90(flipped_cube, -1, axes=(1,2)), np.arange(cube.shape[0]))


plt.plot(criterion)
plt.yscale('log')
plt.show()