import numpy as np
from surfh.Vizualisation import cube_vizualisation
import matplotlib.pyplot as plt

path = '/home/nmonnier/Data/JWST/Point_source/Fusion/Results/lcg_MC_12_MO_4_Temp_None_nit_500_mu_2.50e+01_SD_True/'

# x = np.load(path + 'res_x.npy')
cube = np.load(path + 'res_cube.npy')
criterion = np.load(path + 'criterion.npy')

plt.figure()
plt.plot(criterion)
plt.yscale('log')
# cube_vizualisation.plot_maps(x)
cube_vizualisation.plot_cube(cube, np.arange(cube.shape[0]))
plt.show()