import numpy as np
import os
import udft
from astropy.io import fits
import pathlib



from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.DottestModels import MCMO_SigRLSCT_Model
from surfh.Simulation import simulation_data
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT


def crappy_load_data():

    save_filter_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/'        

    f1 = 'ch1a_00001_corrected_filtered.fits'
    f2 = 'ch1a_00002_corrected_filtered.fits'
    f3 = 'ch1a_00003_corrected_filtered.fits'
    f4 = 'ch1a_00004_corrected_filtered.fits'

    list_data_ch1a = list()
    list_data_ch1b = list()
    list_target_ch1a = list()
    list_target_ch1b = list()
    list_rotation_ch1a = list()
    list_rotation_ch1b = list()
    for file in sorted(os.listdir(save_filter_corrected_dir)):
        if 'ch1a' in file:
            data_shape = (21, 1050, 19)
            with fits.open(save_filter_corrected_dir + file) as hdul:
                    header = hdul[0].header
                    # Add metadata to the header
                    PA_V3a = header['PA_V3'] # Position Angle (V3) in degrees
                    TARG_RA = header['TARG_RA'] # Target Right Ascension (in degrees)
                    TARG_DEC = header['TARG_DEC'] # Target Declination (in degrees)
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1,0,2)
                    list_data_ch1a.append(ndata)
                    list_target_ch1a.append((TARG_RA, TARG_DEC))

        if 'ch1b' in file:
            data_shape = (21, 1213, 19)
            with fits.open(save_filter_corrected_dir + file) as hdul:
                    print(file)
                    header = hdul[0].header
                    # Add metadata to the header
                    PA_V3b = header['PA_V3'] # Position Angle (V3) in degrees
                    TARG_RA = header['TARG_RA'] # Target Right Ascension (in degrees)
                    TARG_DEC = header['TARG_DEC'] # Target Declination (in degrees)
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1,0,2)
                    list_data_ch1b.append(ndata)
                    list_target_ch1b.append((TARG_RA, TARG_DEC))

    
    list_data = list_data_ch1a + list_data_ch1b
    print(list_data)
    #return np.concatenate((np.array(list_data_ch1a).ravel(), np.array(list_data_ch1b).ravel())), list_target_ch1a, list_target_ch1b, PA_V3a, PA_V3b
    return np.array(list_data_ch1b).ravel(), list_target_ch1a, list_target_ch1b, PA_V3a, PA_V3b





"""
Create Model and simulation
"""
sim_dir_path='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, tpl = simulation_data.get_simulation_data(4, 0, sim_dir_path)

# Get indexes of the cube_wavelength for specific wavelength window
indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('1b')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('1b')[-1]))[0]
window_slice = slice(indexes[0]-1, indexes[-1] +1, None) # 

# Update wavelength for simulated data
wavel_axis = wavel_axis[window_slice]

spsf = spsf[window_slice,:,:]
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

templates = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/templates.npy')
templates = templates[:,0:len(wavel_axis)]
print(templates.shape, wavel_axis.shape)


step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

data, list_target_a, list_target_b, rotation_ref_a, rotation_ref_b = crappy_load_data()

grating_resolution_1a = np.mean([3320, 3710])
spec_blur_1a = instru.SpectralBlur(grating_resolution_1a)
# Def Channel spec.
ch1a = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2 + rotation_ref_a),
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
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2 + rotation_ref_b),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1b'),
    name="1B",
)




main_pointing = instru.Coord(0,0)
P1 = main_pointing + instru.Coord(list_target_a[0][0], list_target_a[0][1])
P2 = main_pointing + instru.Coord(list_target_a[1][0], list_target_a[1][1])
P3 = main_pointing + instru.Coord(list_target_a[2][0], list_target_a[2][1])
P4 = main_pointing + instru.Coord(list_target_a[3][0], list_target_a[3][1])
pointings_ch1a = instru.CoordList([P1, P2, P3, P4]).pix(step_Angle.degree)

P1b = main_pointing + instru.Coord(list_target_b[0][0], list_target_b[0][1])
P2b = main_pointing + instru.Coord(list_target_b[1][0], list_target_b[1][1])
P3b = main_pointing + instru.Coord(list_target_b[2][0], list_target_b[2][1])
P4b = main_pointing + instru.Coord(list_target_b[3][0], list_target_b[3][1])
pointings_ch1b = instru.CoordList([P1b, P2b, P3b, P4b]).pix(step_Angle.degree)


alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + np.mean(np.array(list_target_b)[:,0])
beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + np.mean(np.array(list_target_b)[:,1])


pointings = [pointings_ch1b]
spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT_NN(sotf, 
                                              templates, 
                                              alpha_axis, 
                                              beta_axis, 
                                              wavel_axis, 
                                              [ch1b],#, ch3a, ch3b, ch3c], 
                                              step_Angle.degree, 
                                              pointings)

spectroModel.project_FOV()

print("Max Wavel = ", len(wavel_axis))
"""
Reconstruction method
"""
hyperParameter = 5e07
method = "mmmg"
niter = 20
value_init = 1

quadCrit_fusion = fusion_CT.QuadCriterion_MRS(mu_spectro=1, 
                                                    y_spectro=np.copy(data), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)


y_cube = spectroModel.mapsToCube(res_fusion.x)


result_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/'
result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_{len(pointings)}_nit_{str(niter)}_mu_{str(hyperParameter)}/'
path = pathlib.Path(result_path+result_dir)
path.mkdir(parents=True, exist_ok=True)
np.save(path / 'res_x.npy', res_fusion.x)
np.save(path / 'res_cube.npy', y_cube)
np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)
