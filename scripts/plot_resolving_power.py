import numpy as np
import csv
import matplotlib.pyplot as plt
import os 

from surfh.Models import wavelength_mrs
import pathlib



fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resolving_Power/'

# wavelength_1A = wavelength_mrs.get_mrs_wavelength('1A')

# data = np.loadtxt(fusion_path + '1A.csv', delimiter=',')
# wavel = data[:,0]
# power = data[:,1]

# plt.figure()
# plt.plot(wavel, power)
# plt.show()

plt.figure()
for file in sorted(os.listdir(fusion_path)):
    print(pathlib.Path(file).stem)
    data = np.loadtxt(fusion_path + file, delimiter=',')
    SS_wavel = data[:,0]
    SS_power = data[:,1]
    wavelength = wavelength_mrs.get_mrs_wavelength(pathlib.Path(file).stem)
    power_interp = np.interp(wavelength, SS_wavel, SS_power)
    # print(wavelength.shape, power_interp.shape)
#     # np.save(fusion_path+'resolving_power_'+pathlib.Path(file).stem, power_interp)
#     plt.plot(SS_wavel, SS_power, label=file)
#     plt.plot(wavelength, power_interp, label=file + 'interpn')
# plt.legend()
# plt.show()
