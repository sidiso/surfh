import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft

from surfh.Simulation import simulation_data
from surfh.DottestModels import MO_ST_Model
from surfh.ToolsDir import utils

from surfh.Simulation import fusion_CT
from surfh.Models import instru
from surfh.ToolsDir import fusion_mixing

"""
Create Model and simulation
"""
origin_beta_axis, origin_beta_axis, wavel_axis, sotf, maps, tpl = simulation_data.get_simulation_data()

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

# Def Channel spec.
rchan = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=None,
    pce=None,
    wavel_axis=None,
    name="2A",
)

main_pointing = instru.Coord(0, 0)
P1 = instru.Coord((rchan.det_pix_size/3600)/2, rchan.slit_beta_width/2)
# P2 = instru.Coord(-(rchan.det_pix_size/3600)/2, rchan.slit_beta_width/2)
# P3 = instru.Coord((rchan.det_pix_size/3600)/2, -rchan.slit_beta_width/2)
P4 = instru.Coord(-(rchan.det_pix_size/3600)/2, -rchan.slit_beta_width/2)
pointings = instru.CoordList([P1, P4]).pix(step_Angle.degree)

spectroModel = MO_ST_Model.spectroST(sotf=sotf,
                                   templates=tpl,
                                   alpha_axis=origin_beta_axis,
                                   beta_axis=origin_beta_axis,
                                   wavelength_axis=wavel_axis,
                                   instr=rchan,
                                   step_Angle=step_Angle,
                                   pointings=pointings
                                   )

y = spectroModel.forward(maps)
real_cube = spectroModel.mapsToCube(maps)


"""
Reconstruction method
"""
hyperParameter = 0
method = "lcg"
niter = 5
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
utils.plot_3_cube(real_cube, y_adj_cube, y_cube)

plt.figure()
xtick = np.arange(len(quadCrit_fusion.L_crit_val))*5
plt.plot(xtick, quadCrit_fusion.L_crit_val)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()