import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft

from surfh.Simulation import simulation_data
from surfh.DottestModels import MCMO_SigRLSCT_Model
from surfh.ToolsDir import utils

from surfh.Simulation import fusion_CT
from surfh.Models import instru
from surfh.ToolsDir import fusion_mixing

from surfh.Models import wavelength_mrs, realmiri

import pathlib
from pathlib import Path


main_directory = '/home/nmonnier/Data/JWST/Orion_bar/'

fits_directory = main_directory + 'Single_Selected_fits_14062024'
numpy_directory = main_directory + 'Single_numpy_Selected_fits_14062024'
numpy_slices_directory = main_directory + 'Single_slice_numpy_Selected_fits_14062024/'
mask_directory = main_directory + 'Single_mask'

"""
Create Model and simulation
"""
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(4, 40) # subsampling to reduce dim of maps

indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('1a')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('2b')[-1]))[0]
sim_slice = slice(indexes[0], indexes[-1], None) # Slice corresponding to chan 2A

#nwavel_axis = wavel_axis.copy()
wavel_axis = wavel_axis[sim_slice]
templates = templates[:,sim_slice]
spsf = spsf[sim_slice,:,:]
# Select PSF to be the same shape as maps
idx = spsf.shape[1]//2 # Center of the spsf
N = maps.shape[1] # Size of the window
if N%2:
    stepidx = N//2
else:
    stepidx = int(N/2) - 1
start = min(max(idx-stepidx, 0), spsf.shape[1]-N)
#spsf = spsf[:, (100-0):(351+0), (100-0):(351+0)]
spsf = spsf[:, start:start+N, start:start+N]
sotf = udft.ir2fr(spsf, maps.shape[1:])

print("maps shape = ", maps.shape)
print("spsf shape = ", maps.shape)


step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

origin_alpha_width = origin_alpha_axis[-1] - origin_alpha_axis[0]
origin_beta_width = origin_beta_axis[-1] - origin_beta_axis[0]

origin_alpha_width_arcsec = origin_alpha_width*3600
origin_beta_width_arcsec = origin_beta_width*3600

files = []
list_data = []
list_ch1a = []
list_ch1a_data = []
list_ch2a = []
list_ch2a_data = []
for file in os.listdir(fits_directory):
    files.append(file)
    data = np.load(numpy_slices_directory + Path(file).stem + '.npy')
    list_data.append(data)
    
for idx, file in enumerate(files):
    if '1_short' in file:
        list_ch1a.append(file)
        list_ch1a_data.append(list_data[idx])

    if '2_short' in file:
        list_ch2a.append(file)
        list_ch2a_data.append(list_data[idx])

ifu, ref_targ_ra, ref_targ_dec = realmiri.get_IFU(fits_directory + '/' + list_ch1a[0])
# Def Channel spec.
ch1a = instru.IFU(
    fov=ifu.fov,
    det_pix_size=ifu.det_pix_size,
    n_slit=ifu.n_slit,
    w_blur=ifu.w_blur,
    pce=None,
    wavel_axis=ifu.wavel_axis,
    name="1A",
)

pointings1a = []
for file in list_ch1a:
    _, targ_ra, targ_dec = realmiri.get_IFU(fits_directory + '/' + file)
    pointings1a.append(instru.Coord(ref_targ_ra-targ_ra, ref_targ_dec-targ_dec))



ifu, _, _ = realmiri.get_IFU(fits_directory + '/' + list_ch2a[0])
# Def Channel spec.
ch2a = instru.IFU(
    fov=ifu.fov,
    det_pix_size=ifu.det_pix_size,
    n_slit=ifu.n_slit,
    w_blur=ifu.w_blur,
    pce=None,
    wavel_axis=ifu.wavel_axis,
    name="2A",
)
pointings2a = []
for file in list_ch2a:
    _, targ_ra, targ_dec = realmiri.get_IFU(fits_directory + '/' + file)
    pointings2a.append(instru.Coord(ref_targ_ra-targ_ra, ref_targ_dec-targ_dec))

pointings = instru.CoordList(pointings2a).pix(step_Angle.degree)



spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT_NN(sotf, 
                                              templates, 
                                              origin_alpha_axis, 
                                              origin_beta_axis, 
                                              wavel_axis, 
                                              [ch2a],#, ch3a, ch3b, ch3c], 
                                              step_Angle.degree, 
                                              pointings)

# y = np.concatenate((np.asarray(list_ch1a_data).ravel(), np.asarray(list_ch2a_data).ravel()))
y = np.asarray(list_ch2a_data).ravel()
y[np.isnan(y)] = 0
adj = spectroModel.adjoint(y)


"""
Reconstruction method
"""
hyperParameter = 1e9
method = "lcg"
niter = 100
value_init = 1

quadCrit_fusion = fusion_CT.QuadCriterion_MRS(mu_spectro=1, 
                                                    y_spectro=np.copy(y), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)


y_cube = spectroModel.mapsToCube(res_fusion.x)

utils.plot_maps(res_fusion.x)

y_adj = spectroModel.adjoint(y)
y_adj_cube = spectroModel.mapsToCube(y_adj)
utils.plot_3_cube(y_cube, y_cube, y_cube)

plt.figure()
xtick = np.arange(len(quadCrit_fusion.L_crit_val))*5
plt.plot(xtick, quadCrit_fusion.L_crit_val)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()