import numpy as np
import os
import udft
from astropy.io import fits
import pathlib



from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru, spectro_blind_rectangle
from surfh.DottestModels import MCMO_SigRLSCT_Model
from surfh.Simulation import simulation_data
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import criterion_2D

import matplotlib.pyplot as plt
from aljabr import LinOp, dottest


def crappy_load_data():

    save_filter_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/'        

    list_data_ch1c = list()

    list_target_ch1c = list()
    for file in sorted(os.listdir(save_filter_corrected_dir)):

        if 'ch1c' in file:
            data_shape = (21, 1400, 19)
            with fits.open(save_filter_corrected_dir + file) as hdul:
                    print(file)
                    header = hdul[0].header
                    # Add metadata to the header
                    PA_V3c = header['PA_V3'] # Position Angle (V3) in degrees
                    TARG_RA = header['TARG_RA'] # Target Right Ascension (in degrees)
                    TARG_DEC = header['TARG_DEC'] # Target Declination (in degrees)
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1,0,2)
                    list_data_ch1c.append(ndata)
                    list_target_ch1c.append((TARG_RA, TARG_DEC))
    
    return np.array(list_data_ch1c),  list_target_ch1c, PA_V3c

def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


"""
Create Model and simulation
"""
sim_dir_path='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, tpl = simulation_data.get_simulation_data(4, 0, sim_dir_path)

# Get indexes of the cube_wavelength for specific wavelength window
instr_wavelength = wavelength_mrs.get_mrs_wavelength('1c')

choosen_wavelentgh = instr_wavelength[100]
idx_cube_wavelength = find_nearest_idx(wavel_axis, choosen_wavelentgh)


# Update wavelength for simulated data
wavel_axis = wavel_axis[idx_cube_wavelength]
print(f"Instr wavelength = {choosen_wavelentgh} - Closest wavelength = {wavel_axis}")


spsf = spsf[idx_cube_wavelength,:,:]
# Select PSF to be the same shape as maps
idx = spsf.shape[1]//2 # Center of the spsf
N = maps.shape[1] # Size of the window
if N%2:
    stepidx = N//2
else:
    stepidx = int(N/2) - 1
start = min(max(idx-stepidx, 0), spsf.shape[1]-N)
#spsf = spsf[:, (100-0):(351+0), (100-0):(351+0)]
spsf = spsf[start:start+N, start:start+N]

spsf = np.load('/home/nmonnier/Data/JWST/Orion_bar/PSF/psf_1C.npy')
sotf = udft.ir2fr(spsf, maps.shape[1:])


# plt.imshow(spsf)
# plt.colorbar()
# plt.show()
step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

data, list_target_c, rotation_ref_c = crappy_load_data()



grating_resolution_1c = np.mean([3100, 3610])
spec_blur_1c = instru.SpectralBlur(grating_resolution_1c)
# Def Channel spec.
ch1c = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=0),#8.2 + rotation_ref_c),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1c'),
    name="1C",
)

main_pointing = instru.Coord(0,0)
P1c = main_pointing + instru.Coord(list_target_c[0][0] - list_target_c[0][0], list_target_c[0][1] - list_target_c[0][1])
P2c = main_pointing + instru.Coord(list_target_c[1][0] - list_target_c[0][0], list_target_c[1][1] - list_target_c[0][1])
P3c = main_pointing + instru.Coord(list_target_c[2][0] - list_target_c[0][0], list_target_c[2][1] - list_target_c[0][1])
P4c = main_pointing + instru.Coord(list_target_c[3][0] - list_target_c[0][0], list_target_c[3][1] - list_target_c[0][1])
pointings_ch1c = instru.CoordList([P1c, P2c, P3c, P4c]).pix(step_Angle.degree)

# alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + np.mean(np.array(list_target_c)[:,0])
# beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + np.mean(np.array(list_target_c)[:,1])

angle = 8.2 + rotation_ref_c
theta = (angle/180.) * np.pi
rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])


tmp_origin_alpha_axis = (origin_alpha_axis - np.mean(origin_alpha_axis))
tmp_origin_beta_axis = (origin_beta_axis - np.mean(origin_beta_axis))

print("tmp_origin_alpha_axis shape = ", tmp_origin_alpha_axis.shape)
print("rotMatrix ", rotMatrix.shape)

tmp_origin_alpha_axis = (origin_alpha_axis - np.mean(origin_alpha_axis))
conca = np.array([tmp_origin_alpha_axis, tmp_origin_beta_axis])
print(conca.shape)

rot_conca = (rotMatrix @ conca)
print(rot_conca.shape)
rot_alpha_axis = rot_conca[0] - np.mean(rot_conca[0])#(rotMatrix @ tmp_origin_alpha_axis).T
rot_beta_axis = rot_conca[1] - np.mean(rot_conca[1])#(rotMatrix @ tmp_origin_beta_axis).T


alpha_axis = rot_alpha_axis + np.mean(np.array(list_target_c)[:,0])
beta_axis = rot_beta_axis + np.mean(np.array(list_target_c)[:,1])

print(f"Top left coordinate is {alpha_axis[0]}")
print(f'If I calculate = {np.mean(np.array(list_target_c)[:,0]) - step_Angle.degree*len(alpha_axis)//2}')
print(f"{alpha_axis[1]-alpha_axis[0]} - {step_Angle.degree}")
print(((list_target_c[0][0] - list_target_c[0][0])-(list_target_c[1][0] - list_target_c[0][0]))/step_Angle.degree)

alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis)
beta_axis  = origin_beta_axis - np.mean(origin_beta_axis)

main_pointing = instru.Coord(0,0)
P1c = main_pointing + instru.Coord(find_nearest_value(alpha_axis, list_target_c[0][0]- list_target_c[0][0]), 
                                   find_nearest_value(beta_axis, list_target_c[0][1] - list_target_c[0][1]))
P2c = main_pointing + instru.Coord(find_nearest_value(alpha_axis, list_target_c[1][0]- list_target_c[0][0]), 
                                   find_nearest_value(beta_axis, list_target_c[1][1] - list_target_c[0][1]))
P3c = main_pointing + instru.Coord(find_nearest_value(alpha_axis, list_target_c[2][0]- list_target_c[0][0]), 
                                   find_nearest_value(beta_axis, list_target_c[2][1] - list_target_c[0][1]))
P4c = main_pointing + instru.Coord(find_nearest_value(alpha_axis, list_target_c[3][0]- list_target_c[0][0]), 
                                   find_nearest_value(beta_axis, list_target_c[3][1] - list_target_c[0][1]))
pointings_ch1c = instru.CoordList([P1c,P3c]).pix(step_Angle.degree)
print(f'closest value of {list_target_c[2][0]- list_target_c[0][0]} is {find_nearest_value(alpha_axis, list_target_c[2][0]- list_target_c[0][0])}')

spectroModel = spectro_blind_rectangle.MRSBlurred(sotf=sotf,
                                        alpha_axis=alpha_axis,
                                        beta_axis=beta_axis,
                                        instr=ch1c,
                                        step_degree=step_Angle.degree,
                                        pointings=pointings_ch1c)

use_data = data[0::2,:,100,:].ravel()
print("Use data = ", use_data)
print("Use data = ", use_data.shape)
adj = spectroModel.adjoint(use_data)
fw = spectroModel.forward(adj)
adj2 = spectroModel.adjoint(fw)
# plt.figure()
# plt.imshow(adj)
# plt.figure()
# plt.imshow(adj2)
# plt.show()
# adj_mean, adj = spectroModel.data_to_img(use_data)
# plt.figure()
# plt.imshow(np.rot90(np.fliplr(adj),1))
# plt.figure()
# plt.imshow(np.rot90(np.fliplr(adj_mean),1))
# plt.show()
# print(dottest(spectroModel, num=10, echo=True))

"""
Reconstruction method
"""
hyperParameter = 5
method = "qmm"
niter = 100
value_init = 1

quadCrit_fusion = criterion_2D.QuadCriterion_MRS_2D(mu_spectro=1, 
                                                    y_spectro=np.copy(use_data), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)
plt.figure()
plt.plot(quadCrit_fusion.L_crit_val)

plt.figure()
plt.imshow(res_fusion.x)
plt.colorbar()
plt.show()
