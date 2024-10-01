import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft

from surfh.Simulation import simulation_data
from surfh.DottestModels import MO_SigRLSCT_Model
from surfh.ToolsDir import utils

from surfh.Simulation import fusion_CT
from surfh.Models import instru
from surfh.ToolsDir import fusion_mixing

from surfh.Models import wavelength_mrs

import pathlib

"""
Create Model and simulation
"""
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data() # subsampling to reduce dim of maps

indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('2a')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('2a')[-1]))[0]
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

grating_resolution = np.mean([2990, 3110])
spec_blur = instru.SpectralBlur(grating_resolution)

# Def Channel spec.
rchan = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2a'),
    name="2A",
)

main_pointing = instru.Coord(0, 0)
P1 = instru.Coord((rchan.det_pix_size/3600)/4, rchan.slit_beta_width/4)
P2 = instru.Coord(-(rchan.det_pix_size/3600)/4, rchan.slit_beta_width/4)
P3 = instru.Coord((rchan.det_pix_size/3600)/4, -rchan.slit_beta_width/4)
P4 = instru.Coord(-(rchan.det_pix_size/3600)/4, -rchan.slit_beta_width/4)
pointings = instru.CoordList([P1, P2, P3, P4]).pix(step_Angle.degree)

spectroModel = MO_SigRLSCT_Model.spectroSigRLSCT(sotf, 
                                              templates, 
                                              origin_alpha_axis, 
                                              origin_beta_axis, 
                                              wavel_axis, 
                                              rchan, 
                                              step_Angle.degree, 
                                              pointings)

# spectroModel_corrected = MO_SigRLSCT_Model.spectroSigRLSCT_corrected(sotf, 
#                                               templates, 
#                                               origin_alpha_axis, 
#                                               origin_beta_axis, 
#                                               wavel_axis, 
#                                               rchan, 
#                                               step_Angle.degree, 
#                                               pointings)

y = spectroModel.forward(maps)
# yc = spectroModel_corrected(maps)

# adj = spectroModel.adjoint(y)
# adjc = spectroModel_corrected.adjoint(yc)

real_cube = spectroModel.mapsToCube(maps)


"""
Reconstruction method
"""
hyperParameter = 1e7
method = "lcg"
niter = 10
value_init = 0

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
utils.plot_3_cube(real_cube, y_cube, y_cube)

plt.figure()
xtick = np.arange(len(quadCrit_fusion.L_crit_val))*5
plt.plot(xtick, quadCrit_fusion.L_crit_val)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# result_path = '/home/nmonnier/Data/JWST/Orion_bar/fusion_result/simulated/'
# result_dir = f'MO_{len(pointings)}_nit_{str(niter)}_mu_{str(hyperParameter)}/'
# path = pathlib.Path(result_path+result_dir)
# path.mkdir(parents=True, exist_ok=True)
# np.save(path / 'res_x.npy', res_fusion.x)
# np.save(path / 'res_cube.npy', y_cube)
# np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)


# mask = spectroModel.make_mask(maps)

