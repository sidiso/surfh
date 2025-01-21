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
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(4, 0) # subsampling to reduce dim of maps

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

grating_resolution_1a = np.mean([3320, 3710])
spec_blur_1a = instru.SpectralBlur(grating_resolution_1a)
# Def Channel spec.
ch1a = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1a'),
    name="1A",
)

grating_resolution_1b = np.mean([3190, 3750])
spec_blur_1b = instru.SpectralBlur(grating_resolution_1b)
# Def Channel spec.
ch1b = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1b'),
    name="1B",
)

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

grating_resolution_2b = np.mean([2750, 3170])
spec_blur_2b = instru.SpectralBlur(grating_resolution_2b)
ch2b = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2b'),
    name="2B",
)


grating_resolution_2c = np.mean([2860, 3300])
spec_blur_2c = instru.SpectralBlur(grating_resolution_2c)
ch2c = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2c'),
    name="2C",
)

grating_resolution_3a = np.mean([2530, 2880])
spec_blur_3a = instru.SpectralBlur(grating_resolution_3a)
ch3a = instru.IFU(
    fov=instru.FOV(5.5/3600, 6.2/3600, origin=instru.Coord(0, 0), angle=7.5),
    det_pix_size=0.245,
    n_slit=16,
    w_blur=spec_blur_3a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('3a'),
    name="3A",
)

grating_resolution_3b = np.mean([1790, 2640])
spec_blur_3b = instru.SpectralBlur(grating_resolution_3b)
ch3b = instru.IFU(
    fov=instru.FOV(5.5/3600, 6.2/3600, origin=instru.Coord(0, 0), angle=7.5),
    det_pix_size=0.245,
    n_slit=16,
    w_blur=spec_blur_3b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('3b'),
    name="3B",
)

grating_resolution_3c = np.mean([1980, 2790])
spec_blur_3c = instru.SpectralBlur(grating_resolution_3c)
ch3c = instru.IFU(
    fov=instru.FOV(5.5/3600, 6.2/3600, origin=instru.Coord(0, 0), angle=7.5),
    det_pix_size=0.245,
    n_slit=16,
    w_blur=spec_blur_3c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('3c'),
    name="3C",
)



main_pointing = instru.Coord(0,0)
P1 = main_pointing + instru.Coord((ch2a.det_pix_size/3600)/4, ch2a.slit_beta_width/4)
P2 = main_pointing + instru.Coord(-(ch2a.det_pix_size/3600)/4, ch2a.slit_beta_width/4)
P3 = main_pointing + instru.Coord((ch2a.det_pix_size/3600)/4, -ch2a.slit_beta_width/4)
P4 = main_pointing + instru.Coord(-(ch2a.det_pix_size/3600)/4, -ch2a.slit_beta_width/4)
pointings = instru.CoordList([P1]).pix(step_Angle.degree)

spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT_NN(sotf, 
                                              templates, 
                                              origin_alpha_axis, 
                                              origin_beta_axis, 
                                              wavel_axis, 
                                              [ch1a],#, ch3a, ch3b, ch3c], 
                                              step_Angle.degree, 
                                              pointings)

y = spectroModel.forward(maps)
adj = spectroModel.adjoint(y)
real_cube = spectroModel.mapsToCube(maps)
a = spectroModel.channels[0].sliceToCube(y)




fig, (ax1, ax2) = plt.subplots(1,2)
img1 = ax1.imshow(real_cube[spectroModel.channels[0].wslice][-1])
img2 = ax2.imshow(a[-1])

ticks = np.linspace(0,251,5)

ticks_slice = slice(0,251,50)
nexticks = origin_alpha_axis[ticks_slice]*3600
ticklabels = ["{:6.2f}".format(i) for i in nexticks]
ticks = np.linspace(0,251,len(nexticks), dtype=int)


x_label_list = ["{:6.2f}".format(origin_alpha_axis[20]*3600), 
                "{:6.2f}".format(origin_alpha_axis[70]*3600), 
                "{:6.2f}".format(origin_alpha_axis[120]*3600), 
                "{:6.2f}".format(origin_alpha_axis[170]*3600), 
                "{:6.2f}".format(origin_alpha_axis[220]*3600)]

y_label_list = ["{:6.2f}".format(origin_beta_axis[20]*3600), 
                "{:6.2f}".format(origin_beta_axis[70]*3600), 
                "{:6.2f}".format(origin_beta_axis[120]*3600), 
                "{:6.2f}".format(origin_beta_axis[170]*3600), 
                "{:6.2f}".format(origin_beta_axis[220]*3600)]

ax1.set_xticks([20,70,120,170, 220])
ax1.set_xticklabels(x_label_list, rotation=90, size=12)
ax1.set_yticks([20,70,120,170, 220])
ax1.set_yticklabels(x_label_list, size=12)

ax2.set_xticks([20,70,120,170, 220])
ax2.set_xticklabels(x_label_list, rotation=90, size=12)
ax2.set_yticks([20,70,120,170, 220])
ax2.set_yticklabels(x_label_list, size=12)

ax1.set_title(f"Real data $\lambda = {wavel_axis[spectroModel.channels[0].wslice][-1]}$", size=25)
ax2.set_title(f"Simulated data $\lambda = {wavel_axis[spectroModel.channels[0].wslice][-1]}$", size=25)
plt.show()


mask = np.array(spectroModel.make_small_mask())
mask[np.where(mask != 0)] = 1
path = '/home/nmonnier/Data/JWST/Orion_bar/fusion_result/simulated/MC_5_MO_4_nit_2000_mu_10000000.0_SS_4'
res_cube = np.load(path + '/res_cube.npy')
masked_res_cube = res_cube*mask[0]
masked_real_cube = real_cube*mask[0]



plt.figure()
plt.plot(wavel_axis, np.mean(masked_real_cube, axis=(1,2)), label='Real Data')
plt.plot(wavel_axis, np.mean(masked_res_cube, axis=(1,2)), label='Reconstruction')
plt.legend(fontsize=12)
plt.xlabel('Wavelength $\lambda$', fontsize=20)
plt.show()



diff_cube = np.zeros_like(maps)
diff_cube[0] = 100*(masked_real_cube[0] - masked_res_cube[0])/masked_real_cube[0] # Lamnda = 4.9
diff_cube[1] = 100*(masked_real_cube[541] - masked_res_cube[541])/masked_real_cube[541] # Lamnda = 6.42
diff_cube[2] = 100*(masked_real_cube[1000] - masked_res_cube[1000])/masked_real_cube[1000] # Lamnda = 8.07
diff_cube[3] = 100*(masked_real_cube[1449] - masked_res_cube[1449])/masked_real_cube[1449] # Lamnda = 10.11
utils.plot_maps(diff_cube)
plt.show()

"""
Reconstruction method
"""
# hyperParameter = 5e5
# method = "lcg"
# niter = 1000
# value_init = 1

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


# result_path = '/home/nmonnier/Data/JWST/Orion_bar/fusion_result/simulated/'
# result_dir = f'MC_{len(spectroModel.instrs)}_MO_{len(pointings)}_nit_{str(niter)}_mu_{str(hyperParameter)}/'
# path = pathlib.Path(result_path+result_dir)
# path.mkdir(parents=True, exist_ok=True)
# np.save(path / 'res_x.npy', res_fusion.x)
# np.save(path / 'res_cube.npy', y_cube)
# np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)

# plt.show()

# mask = spectroModel.make_mask(maps)



