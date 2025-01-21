import matplotlib.pyplot as plt
import numpy as np

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

from surfh.Models import wavelength_mrs

import pathlib

"""
Create Model and simulation
"""
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(3) # subsampling to reduce dim of maps

indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('1c')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('2c')[-1]))[0]
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



step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

origin_alpha_width = origin_alpha_axis[-1] - origin_alpha_axis[0]
origin_beta_width = origin_beta_axis[-1] - origin_beta_axis[0]

origin_alpha_width_arcsec = origin_alpha_width*3600
origin_beta_width_arcsec = origin_beta_width*3600


grating_resolution_1c = np.mean([3100, 3610])
spec_blur_1c = instru.SpectralBlur(grating_resolution_1c)
# Def Channel spec.
ch1c = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1c'),
    name="1C",
)

grating_resolution_2a = np.mean([2990, 3110])
spec_blur_2a = instru.SpectralBlur(grating_resolution_2a)
ch2a = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2a'),
    name="2A",
)

grating_resolution_2b = np.mean([2990, 3110])
spec_blur_2a = instru.SpectralBlur(grating_resolution_2b)
ch2b = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2b'),
    name="2B",
)

grating_resolution_2c = np.mean([2990, 3110])
spec_blur_2c = instru.SpectralBlur(grating_resolution_2c)
ch2c = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2c'),
    name="2C",
)

# grating_resolution_3a = np.mean([2990, 3110])
# spec_blur_3a = instru.SpectralBlur(grating_resolution_3a)
# ch3a = instru.IFU(
#     fov=instru.FOV(5.5/3600, 6.2/3600, origin=instru.Coord(0, 0), angle=7.5),
#     det_pix_size=0.245,
#     n_slit=16,
#     w_blur=spec_blur_3a,
#     pce=None,
#     wavel_axis=wavelength_mrs.get_mrs_wavelength('3a'),
#     name="3A",
# )


main_pointing = instru.Coord((ch2a.det_pix_size/3600)*7, -(ch2a.det_pix_size/3600)*7)
P1 = main_pointing + instru.Coord((ch2a.det_pix_size/3600)/4, ch2a.slit_beta_width/4)
P2 = main_pointing + instru.Coord(-(ch2a.det_pix_size/3600)/4, ch2a.slit_beta_width/4)
P3 = main_pointing + instru.Coord((ch2a.det_pix_size/3600)/4, -ch2a.slit_beta_width/4)
P4 = main_pointing + instru.Coord(-(ch2a.det_pix_size/3600)/4, -ch2a.slit_beta_width/4)
pointings = instru.CoordList([P1, P2, P3, P4]).pix(step_Angle.degree)

spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT(sotf, 
                                              templates, 
                                              origin_alpha_axis, 
                                              origin_beta_axis, 
                                              wavel_axis, 
                                              [ch1c, ch2a, ch2b, ch2c], 
                                              step_Angle.degree, 
                                              pointings)
print("FW")
y = spectroModel.forward(maps)
# print("ADJ")
adj = spectroModel.adjoint(y)
real_cube = spectroModel.mapsToCube(maps)


y2 = spectroModel.forward(adj)
# print("ADJ")
adj2 = spectroModel.adjoint(y2)
utils.plot_maps(adj)
utils.plot_maps(adj2)
plt.show()


# # np.save('reference_NN_MCMO_SigRLSCT_Simulated_fw.npy', y)
# # np.save('reference_NN_MCMO_SigRLSCT_Simulated_ad.npy', adj)
# utils.plot_maps(adj)
# plt.show()
# y_ref = np.load('reference_NN_MCMO_SigRLSCT_Simulated_fw.npy')
# adj_ref = np.load('reference_NN_MCMO_SigRLSCT_Simulated_ad.npy')

# print(np.allclose(y, y_ref))
# print(np.allclose(adj, adj_ref))
# """
# Reconstruction method
# """
# hyperParameter = 1e6
# method = "lcg"
# niter = 5
# value_init = 0

# quadCrit_fusion = fusion_CT.QuadCriterion_MRS(mu_spectro=1, 
#                                                     y_spectro=np.copy(y), 
#                                                     model_spectro=spectroModel, 
#                                                     mu_reg=hyperParameter, 
#                                                     printing=True, 
#                                                     gradient="separated"
#                                                     )

# res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)


# y_cube = spectroModel.mapsToCube(res_fusion.x)

# utils.plot_maps(res_fusion.x)

# y_adj = spectroModel.adjoint(y)
# y_adj_cube = spectroModel.mapsToCube(y_adj)
# utils.plot_3_cube(real_cube, y_cube, y_cube)

# plt.figure()
# xtick = np.arange(len(quadCrit_fusion.L_crit_val))*5
# plt.plot(xtick, quadCrit_fusion.L_crit_val)
# plt.yscale("log")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)


# # result_path = '/home/nmonnier/Data/JWST/Orion_bar/fusion_result/simulated/'
# # result_dir = f'NN_True_MC_{len(spectroModel.instrs)}_MO_{len(pointings)}_nit_{str(niter)}_mu_{str(hyperParameter)}/'
# # path = pathlib.Path(result_path+result_dir)
# # path.mkdir(parents=True, exist_ok=True)
# # np.save(path / 'res_x.npy', res_fusion.x)
# # np.save(path / 'res_cube.npy', y_cube)
# # np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)


plt.show()



