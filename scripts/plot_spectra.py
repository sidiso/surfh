import matplotlib.pyplot as plt
import numpy as np


with open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/mean_flux_fusion.dat', 'r') as file1:
    content = file1.read()
    arr_cont = np.array(content)
    print(arr_cont)


fusion = np.loadtxt('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/mean_flux_fusion.dat', delimiter=" ", unpack=False)
real = np.loadtxt('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/mean_flux_real.dat', delimiter=" ", unpack=False)

plt.figure()
plt.plot(fusion[:,0], fusion[:,1], label='Fusion', linewidth=3)
plt.plot(real[:,0], real[:,1], label='Pipeline', linewidth=3)
plt.legend(fontsize="20")
plt.show()

plt.figure()
plt.plot(fusion[:,0], (fusion[:,1]-real[:,1])/fusion[:,1], label='Fusion', linewidth=3)
plt.show()

