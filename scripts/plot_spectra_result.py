import numpy as np
import os
import udft
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import pathlib
import matplotlib.pyplot as plt


main_dir = 'lcg_MC_12_MO_4_Temp_6_nit_200_mu_1.00e+03_SD_True/'

fusion_dir_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/'

ref_dir_path = fusion_dir_path + 'Reference/'
result_dir_path = fusion_dir_path + 'Results/'
template_dir_path = fusion_dir_path + 'Templates/'

fusion = np.loadtxt(result_dir_path + main_dir + 'mean_flux_fusion.dat', delimiter=" ", unpack=False)
real = np.loadtxt(ref_dir_path + 'mean_flux_real.dat', delimiter=" ", unpack=False)
fusion_wavel = np.load(template_dir_path + 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_6_templates_SS4.npy')

hdul = fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits')
real_wavel = np.array(hdul[5].data[0])[0,:,0]



plt.figure()
plt.plot(fusion_wavel, fusion[:,1], label='Fusion', linewidth=3)
plt.plot(real_wavel, real[:,1], label='Pipeline', linewidth=3)
plt.legend(fontsize="20")
plt.title("Mean spectra", fontsize="32")
plt.show()
