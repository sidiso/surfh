import numpy as np
import os
import udft
from astropy.io import fits
import pathlib
from scipy.ndimage import rotate



from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru, spectro_blind
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
print("origin alpha axis is ", origin_alpha_axis)

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)


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
spsf = spsf/np.sum(spsf)
sotf = udft.ir2fr(spsf, maps.shape[1:])


print(f'sum psf = {np.sum(spsf)}')

# plt.figure()
# plt.imshow(spsf)
# plt.colorbar()
# plt.figure()
# plt.imshow(np.log10(spsf))
# plt.colorbar()
# plt.show()
# angle = 67.35  # Replace with the desired angle in degrees

# # Rotate the array around its center
# rotated_array = rotate(spsf, angle, reshape=False, order=3)

# # Display the original and rotated arrays
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(np.log10(spsf), cmap='gray')
# axes[0].set_title('Original Array')
# axes[1].imshow(np.log10(rotated_array), cmap='gray')
# axes[1].set_title(f'Rotated Array by {angle} degrees')
# plt.show()

data, list_target_c, rotation_ref_c = crappy_load_data()


mixed_maps = (0.4*maps[0] + 0.5*maps[1] + 0.4*maps[2] + 0.3*maps[3])*10000

grating_resolution_1c = np.mean([3100, 3610])
spec_blur_1c = instru.SpectralBlur(grating_resolution_1c)
# Def Channel spec.
ch1c = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2 - rotation_ref_c),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1c'),
    name="1C",
)

main_pointing = instru.Coord(0,0)
P1c = main_pointing + instru.Coord(list_target_c[0][0], list_target_c[0][1])
P2c = main_pointing + instru.Coord(list_target_c[1][0], list_target_c[1][1])
P3c = main_pointing + instru.Coord(list_target_c[2][0], list_target_c[2][1])
P4c = main_pointing + instru.Coord(list_target_c[3][0], list_target_c[3][1])
pointings_ch1c = instru.CoordList([P1c, P2c, P3c, P4c]).pix(step_Angle.degree)

alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + np.mean(np.array(list_target_c)[:,0])
beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + np.mean(np.array(list_target_c)[:,1])

spectroModel = spectro_blind.MRSBlurred(sotf=sotf,
                                        alpha_axis=alpha_axis,
                                        beta_axis=beta_axis,
                                        instr=ch1c,
                                        step_degree=step_Angle.degree,
                                        pointings=pointings_ch1c)

simulated_data = spectroModel.forward(np.fliplr(mixed_maps))
# adj = spectroModel.adjoint(simulated_data)
# plt.figure()
# plt.imshow(adj)
# plt.figure()
# plt.imshow(np.fliplr(mixed_maps))
# plt.show()

# adj_mean, adj = spectroModel.data_to_img(simulated_data)
# plt.figure()
# plt.imshow(np.rot90(np.fliplr(adj),1))
# plt.figure()
# plt.imshow(np.flipud(np.rot90(np.fliplr(adj_mean),1)))
# plt.show()
# print(dottest(spectroModel, num=10, echo=True))

"""
Reconstruction method
"""
hyperParameter = 5
method = "lcg"
niter = 600
value_init = 0

quadCrit_fusion = criterion_2D.QuadCriterion_MRS_2D(mu_spectro=1, 
                                                    y_spectro=np.copy(simulated_data), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)
plt.figure()
plt.plot(np.arange(len(quadCrit_fusion.L_crit_val))*5, quadCrit_fusion.L_crit_val)
plt.yscale("log")

adj_mean, adj = spectroModel.data_to_img(np.copy(simulated_data))
plt.figure()
plt.title("Projected data")
plt.imshow(adj_mean)
plt.colorbar()
plt.figure()
plt.title("Real data")
plt.imshow(np.fliplr(mixed_maps))
plt.colorbar()

plt.figure()
plt.title("Deconv data")
plt.imshow(res_fusion.x)
plt.colorbar()
plt.show()
