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
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT
from aljabr import LinOp, dottest
from scipy import ndimage


def crappy_load_data():

    save_filter_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/'        

    list_data_ch1a = list()
    list_data_ch1b = list()
    list_data_ch1c = list()

    list_target_ch1a = list()
    list_target_ch1b = list()
    list_target_ch1c = list()
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
    
    list_data = list_data_ch1a + list_data_ch1b
    #return np.concatenate((np.array(list_data_ch1a).ravel(), np.array(list_data_ch1b).ravel())), list_target_ch1a, list_target_ch1b, PA_V3a, PA_V3b
    # return np.array(list_data_ch1b).ravel(), list_target_ch1a, list_target_ch1b, PA_V3a, PA_V3b
    return np.array(list_data_ch1c).ravel(), list_target_ch1b, list_target_ch1c, PA_V3b, PA_V3c





"""
Create Model and simulation
"""
psf_dir_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/PSF/'
sim_dir_path='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
_, _, wavel_axis, _, _, _ = simulation_data.get_simulation_data(4, 0, sim_dir_path)

# Get indexes of the cube_wavelength for specific wavelength window
indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('1c')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('1c')[-1]))[0]
window_slice = slice(indexes[0]-1, indexes[-1] +1, None) # 

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

N = 301
imshape = (N, N)
origin_alpha_axis = np.arange(imshape[0]) * step_Angle.degree
origin_beta_axis = np.arange(imshape[1]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)

# Update wavelength for simulated data
wavel_axis = wavel_axis[window_slice]
wavel_axis = wavelength_mrs.get_mrs_wavelength('1c')

templates = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Templates/nmf_orion_1C_2_templates_1400samples.npy')
if len(templates.shape) == 1:
      templates = templates[np.newaxis,...]
print(templates.shape, wavel_axis.shape)

" If SubSampled wavelength"
# spsf = np.load(psf_dir_path + 'psfs_pixscale0.025_npix_301_fov7.525.npy')
# spsf = spsf[window_slice]
# sotf = udft.ir2fr(spsf, imshape)

" If same Wavelength "
spsf = np.load(psf_dir_path + 'psfs_pixscale0.025_npix_301_fov7.525_chan_1C.npy')
sotf = udft.ir2fr(spsf, imshape)


step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

data, list_target_b, list_target_c, rotation_ref_b, rotation_ref_c = crappy_load_data()

# grating_resolution_1a = np.mean([3320, 3710])
# spec_blur_1a = instru.SpectralBlur(grating_resolution_1a)
# # Def Channel spec.
# ch1a = instru.IFU(
#     fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2 + rotation_ref_a),
#     det_pix_size=0.196,
#     n_slit=21,
#     w_blur=spec_blur_1a,
#     pce=None,
#     wavel_axis=wavelength_mrs.get_mrs_wavelength('1a'),
#     name="1A",
# )

grating_resolution_1b = np.mean([3190, 3750])
spec_blur_1b = instru.SpectralBlur(grating_resolution_1b)
# Def Channel spec.
ch1b = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2 - rotation_ref_b),
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
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle=8.2 - rotation_ref_c),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1c'),
    name="1C",
)



main_pointing = instru.Coord(0,0)
# P1 = main_pointing + instru.Coord(list_target_a[0][0], list_target_a[0][1])
# P2 = main_pointing + instru.Coord(list_target_a[1][0], list_target_a[1][1])
# P3 = main_pointing + instru.Coord(list_target_a[2][0], list_target_a[2][1])
# P4 = main_pointing + instru.Coord(list_target_a[3][0], list_target_a[3][1])
# pointings_ch1a = instru.CoordList([P1, P2, P3, P4]).pix(step_Angle.degree)

P1b = main_pointing + instru.Coord(list_target_b[0][0], list_target_b[0][1])
P2b = main_pointing + instru.Coord(list_target_b[1][0], list_target_b[1][1])
P3b = main_pointing + instru.Coord(list_target_b[2][0], list_target_b[2][1])
P4b = main_pointing + instru.Coord(list_target_b[3][0], list_target_b[3][1])
pointings_ch1b = instru.CoordList([P1b, P2b, P3b, P4b]).pix(step_Angle.degree)

P1c = main_pointing + instru.Coord(list_target_c[0][0], list_target_c[0][1])
P2c = main_pointing + instru.Coord(list_target_c[1][0], list_target_c[1][1])
P3c = main_pointing + instru.Coord(list_target_c[2][0], list_target_c[2][1])
P4c = main_pointing + instru.Coord(list_target_c[3][0], list_target_c[3][1])
pointings_ch1c = instru.CoordList([P1c, P2c, P3c, P4c]).pix(step_Angle.degree)


alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + np.mean(np.array(list_target_c)[:,0])
beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + np.mean(np.array(list_target_c)[:,1])

pointings = [pointings_ch1c]

spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT(sotf=sotf, 
                                                    templates=templates, 
                                                    alpha_axis=alpha_axis, 
                                                    beta_axis=beta_axis, 
                                                    wavelength_axis=wavel_axis, 
                                                    instrs=[ch1c],#, ch3a, ch3b, ch3c], 
                                                    step_degree=step_Angle.degree, 
                                                    pointings=pointings)

use_data = spectroModel.real_data_janskySR_to_jansky(data)

# print(dottest(spectroModel, num=10, echo=True))

# cube = spectroModel.adjoint(data)
# cube_vizualisation.plot_cube(cube, spectroModel.wavelength_axis)


# cube_list, wave_list = spectroModel.test_vizual_projection(data)
# cube_vizualisation.plot_concatenated_cubes(cube_list, wave_list)
# spectroModel.project_FOV()

from astropy.wcs import WCS
from astropy.table import Table

with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/ChannelCube_ch1-long_s3d.fits') as hdul:
      hdr = hdul[1].header
      RA = hdr['CRVAL1']
      DEC = hdr['CRVAL2']
origin_alpha_axis = np.arange(imshape[0]) * step_Angle.degree
origin_beta_axis = np.arange(imshape[1]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)
alpha_axis = origin_alpha_axis + RA
beta_axis = origin_beta_axis + DEC



# from scipy.ndimage import rotate

# cube_data = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/lcg_MC_1_MO_1_nit_32_mu_500000000.0/res_cube.npy')
# mask = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Masks/binary_mask_1C.npy')
# masked_fusion_cube = cube_data*mask[np.newaxis,...]
# flipped_cube = np.array([np.rot90(np.fliplr(masked_fusion_cube[i]), 1) for i in range(masked_fusion_cube.shape[0])])


# angle = 8.2  # angle en degrés

# # Créer un nouveau cube pour stocker les résultats
# rotated_cube = np.empty_like(flipped_cube)

# # Appliquer la rotation pour chaque tranche spectrale
# for i in range(flipped_cube.shape[0]):
#     # Appliquer la rotation en 2D en gardant le centre de la tranche comme centre de rotation
#     rotated_cube[i] = rotate(flipped_cube[i], angle=angle, reshape=False, axes=(1, 0), order=3, mode='nearest')




# # Créer l'objet WCS avec les dimensions et les types de coordonnées
# # Création de l'objet WCS pour définir les métadonnées globales
# wcs = WCS(naxis=3)
# wcs.wcs.crpix = [1, 1, 1]  # Point de référence de chaque axe (pixel 1,1,1)
# wcs.wcs.crval = [alpha_axis[0], beta_axis[0], wavel_axis[0]]  # Valeurs de départ pour chaque axe
# wcs.wcs.cdelt = [np.diff(alpha_axis).mean(), np.diff(beta_axis).mean(), np.diff(wavel_axis).mean()]  # Résolution moyenne pour chaque axe
# wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]  # Types d'axes : longueur d'onde, RA, et DEC

# rotated_cube[rotated_cube < 100] = np.nan
# # Créer le PrimaryHDU avec les données et les informations WCS
# hdu_data = fits.PrimaryHDU(data=rotated_cube, header=wcs.to_header())

# # Création d'extensions pour sauvegarder les valeurs exactes des axes
# # Table pour les longueurs d'onde (axe spectral)
# wavel_table = Table([wavel_axis], names=['WAVELENGTH'])
# hdu_wavel = fits.BinTableHDU(wavel_table, name='WAVEL_AXIS')

# # Table pour les coordonnées spatiales RA (alpha) et DEC (beta)
# alpha_beta_table = Table([alpha_axis, beta_axis], names=['ALPHA_RA', 'BETA_DEC'])
# hdu_alpha_beta = fits.BinTableHDU(alpha_beta_table, name='SPATIAL_AXES')

# # Sauvegarder dans un fichier FITS
# hdul = fits.HDUList([hdu_data, hdu_wavel, hdu_alpha_beta])
# output_filename = "/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/datacube_with_custom_axes.fits"
# hdul.writeto(output_filename, overwrite=True)

# print(f"Fichier FITS '{output_filename}' créé avec succès, contenant le datacube et les axes personnalisés.")




print("Max Wavel = ", len(wavel_axis))
"""
Reconstruction method
"""
hyperParameter = 5e8
method = "lcg"
niter = 33
value_init = 0

quadCrit_fusion = fusion_CT.QuadCriterion_MRS(mu_spectro=1, 
                                                    y_spectro=np.copy(data), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)


y_cube = spectroModel.mapsToCube(res_fusion.x)
flipped_cube = np.array([np.fliplr(y_cube[i]) for i in range(y_cube.shape[0])])


result_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Results/'
result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_{len(pointings)}_nit_{str(niter)}_mu_{str(hyperParameter)}/'
path = pathlib.Path(result_path+result_dir)
path.mkdir(parents=True, exist_ok=True)
np.save(path / 'res_x.npy', res_fusion.x)
np.save(path / 'res_cube.npy', y_cube)
np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)


cube_vizualisation.plot_cube(np.rot90(flipped_cube, -1, axes=(1,2)), wavel_axis)
