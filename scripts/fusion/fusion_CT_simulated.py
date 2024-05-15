import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft

from surfh.Simulation import simulation_data
from surfh.DottestModels import CT_Model
from surfh.ToolsDir import utils

from surfh.Simulation import fusion_CT
from surfh.ToolsDir import fusion_mixing

"""
Create Model and simulation
"""
origin_beta_axis, origin_beta_axis, wavel_axis, sotf, maps, tpl = simulation_data.get_simulation_data()

spectroModel = CT_Model.CT_spectro(sotf=sotf,
                                   templates=tpl,
                                   alpha_axis=origin_beta_axis,
                                   beta_axis=origin_beta_axis,
                                   wavelength_axis=wavel_axis)

y = spectroModel.forward(maps)

y_maps = spectroModel.cubeTomaps(y)
real_cube = spectroModel.mapsToCube(maps)


"""
Reconstruction method
"""
hyperParameter = 1e5
method = "lcg"
niter = 1000
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

utils.plot_3_cube(real_cube, y, y_cube)

plt.figure()
xtick = np.arange(len(quadCrit_fusion.L_crit_val))*5
plt.plot(xtick, quadCrit_fusion.L_crit_val)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()