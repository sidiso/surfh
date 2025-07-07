import numpy as np
from surfh.Vizualisation import cube_vizualisation


path = '/home/nmonnier/Data/JWST/Point_source/Fusion/Results/lcg_MC_12_MO_4_Temp_6_nit_5_mu_1.00e+00_SD_False/'

x = np.load(path + 'res_x.npy')
cube = np.load(path + 'res_cube.npy')
criterion = np.load(path + 'criterion.npy')


cube_vizualisation.plot_cube(cube, np.arange(cube.shape[0]))