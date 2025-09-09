import numpy as np
from surfh.Vizualisation import cube_vizualisation
import matplotlib.pyplot as plt

path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_11_MO_4_Temp_14_nit_150_mu_5.00e+06_SD_True/'

# wavel = np.load(path + 'wavel.npy')
x = np.load(path + 'res_x.npy')
# cube = np.load(path + 'res_cube.npy')
criterion = np.load(path + 'criterion.npy')
# spectre = np.load('/home/nmonnier/Data/JWST/Point_source/Fusion/Templates/nmf_SN2023fyq_2ABC_3ABC_6_templates_SS4.npy')

# plt.figure()
# for i in range(spectre.shape[0]):
#     plt.plot(spectre[i], label=i)
# plt.legend()
plt.figure()
plt.plot(criterion[3:])
plt.yscale('log')
cube_vizualisation.plot_maps(x)
# cube_vizualisation.plot_cube(cube, np.arange(cube.shape[0]))
plt.show()