import os
import numpy as np
import matplotlib.pyplot as plt

from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.ToolsDir import utils

res_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/lcg_MC_6_MO_4_Temp_6_nit_450_mu_5.00e+09/'

cube = np.load(res_dir + 'res_cube.npy')
maps = np.load(res_dir + 'res_x.npy')
criterion = np.load(res_dir + 'criterion.npy')

# utils.plot_maps(maps)

cube_vizualisation.plot_cube(cube, np.arange(cube.shape[0]))

plt.plot(criterion)
plt.yscale('log')
plt.show()