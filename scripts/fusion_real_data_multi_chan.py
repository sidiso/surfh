import numpy as np
import os
import udft
from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt

from rich import print
from rich.progress import track
from rich.console import Console

from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.DottestModels import MCMO_SigRLSCT_Model
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT
from aljabr import LinOp, dottest
from scipy import ndimage

console = Console()


def load_data(list_chan, save_filter_corrected_dir):

    data_dict = {}
    data_dict['data'] = {}
    data_dict["target"] = {}
    data_dict["rotation"] = {}
    # Init main dict
    for chan in list_chan:
        data_dict['data'][chan] = list()
        data_dict["target"][chan] = list()
        data_dict["rotation"][chan] = 0.

    datashape = {}
    datashape['1a'] = (21, 1050, 19)
    datashape['1b'] = (21, 1213, 19)
    datashape['1c'] = (21, 1400, 19)
    datashape['2a'] = (17, 970, 24)
    datashape['2b'] = (17, 1124, 24)
    datashape['2c'] = (17, 1300, 24)
    datashape['3a'] = (16, 769, 24)
    datashape['3b'] = (16, 892, 24)
    datashape['3c'] = (16, 1028, 24)
    datashape['4a'] = (12, 542, 27)
    datashape['4b'] = (12, 632, 27)
    datashape['4c'] = (12, 717, 27)
    
    for file in sorted(os.listdir(save_filter_corrected_dir)):
        for chan in list_chan:
            if chan in file:
                data_shape = datashape[chan] # TODO get shape regarding chan/band
                with fits.open(save_filter_corrected_dir + file) as hdul:
                    header = hdul[0].header
                    # Add metadata to the header
                    PA_V3 = header['PA_V3'] # Position Angle (V3) in degrees
                    TARG_RA = header['TARG_RA'] # Target Right Ascension (in degrees)
                    TARG_DEC = header['TARG_DEC'] # Target Declination (in degrees)
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1,0,2)

                    data_dict['data'][chan].append(ndata)
                    data_dict['target'][chan].append((TARG_RA, TARG_DEC))
                    data_dict['rotation'][chan] = PA_V3

    return  data_dict



"""
Create Model and simulation
"""


fusion_dir_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/'

psf_dir_path = fusion_dir_path + 'PSF/'
template_dir_path = fusion_dir_path +'Templates/'
save_filter_corrected_dir = fusion_dir_path + 'Filtered_slices/'        
result_path = fusion_dir_path + '/Results/'

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

N = 501
imshape = (N, N)
origin_alpha_axis = np.arange(imshape[0]) * step_Angle.degree
origin_beta_axis = np.arange(imshape[1]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)

wavel_axis = np.load(template_dir_path + 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy')
templates = np.load(template_dir_path + 'nmf_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy')
spsf = np.load(psf_dir_path + 'psfs_pixscale0.025_npix_501_fov12.525_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy')

sotf = udft.ir2fr(spsf, imshape)





print("tempaltes shape = ", templates.shape)
if len(templates.shape) == 1:
      templates = templates[np.newaxis,...]

templates = templates/10e3

# data, list_target_b, list_target_c, rotation_ref_b, rotation_ref_c = crappy_load_data()

list_chan = ['1a', '1b', '1c' , '2a' , '2b' , '2c', '3a', '3b', '3c', '4a', '4b', '4c']
console.log("[bold cyan]Loading data...[/bold cyan]")
data_dict = load_data(list_chan, save_filter_corrected_dir)
console.log("[green]Data loaded successfully![/green]")



data = list()
for chan in list_chan:
    data.append(np.array(data_dict['data'][chan]).ravel())
ndata = np.concatenate(data)

grating_resolution_1a = np.mean([3320, 3710])
spec_blur_1a = instru.SpectralBlur(grating_resolution_1a)
# Def Channel spec.
ch1a = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['1a']),
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
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['1b']),
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
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['1c']),
    det_pix_size=0.196,
    n_slit=21,
    w_blur=spec_blur_1c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('1c'),
    name="1C",
)

grating_resolution_2a = np.mean([2990, 3110])
spec_blur_2a = instru.SpectralBlur(grating_resolution_2a)
# Def Channel spec.
ch2a = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['2a']),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2a'),
    name="2A",
)


grating_resolution_2b = np.mean([2750, 3170])
spec_blur_2b = instru.SpectralBlur(grating_resolution_2b)
# Def Channel spec.
ch2b = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['2b']),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2b'),
    name="2B",
)

grating_resolution_2c = np.mean([2860, 3300])
spec_blur_2c = instru.SpectralBlur(grating_resolution_2c)
# Def Channel spec.
ch2c = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['2c']),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur_2c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('2c'),
    name="2C",
)

grating_resolution_3a = np.mean([2530, 2880])
spec_blur_3a = instru.SpectralBlur(grating_resolution_3a)
# Def Channel spec.
ch3a = instru.IFU(
    fov=instru.FOV(5.2/3600, 6.2/3600, origin=instru.Coord(0, 0), angle=- data_dict["rotation"]['3a']),
    det_pix_size=0.245,
    n_slit=16,
    w_blur=spec_blur_3a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('3a'),
    name="3A",
)


grating_resolution_3b = np.mean([1790, 2640])
spec_blur_3b = instru.SpectralBlur(grating_resolution_3b)
# Def Channel spec.
ch3b = instru.IFU(
    fov=instru.FOV(5.2/3600, 6.2/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['3b']),
    det_pix_size=0.245,
    n_slit=16,
    w_blur=spec_blur_3b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('3b'),
    name="3B",
)

grating_resolution_3c = np.mean([1980, 2790])
spec_blur_3c = instru.SpectralBlur(grating_resolution_3c)
# Def Channel spec.
ch3c = instru.IFU(
    fov=instru.FOV(5.2/3600, 6.2/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['3c']),
    det_pix_size=0.245,
    n_slit=16,
    w_blur=spec_blur_3c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('3c'),
    name="3C",
)

grating_resolution_4a = np.mean([1460, 1930])
spec_blur_4a = instru.SpectralBlur(grating_resolution_4a)
# Def Channel spec.
ch4a = instru.IFU(
    fov=instru.FOV(6.6/3600, 7.7/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['4a']),
    det_pix_size=0.273,
    n_slit=12,
    w_blur=spec_blur_4a,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('4a'),
    name="4A",
)

grating_resolution_4b = np.mean([1680, 1760])
spec_blur_4b = instru.SpectralBlur(grating_resolution_4b)
# Def Channel spec.
ch4b = instru.IFU(
    fov=instru.FOV(6.6/3600, 7.7/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['4b']),
    det_pix_size=0.273,
    n_slit=12,
    w_blur=spec_blur_4b,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('4b'),
    name="4B",
)

grating_resolution_4c = np.mean([1630, 1330])
spec_blur_4c = instru.SpectralBlur(grating_resolution_4c)
# Def Channel spec.
ch4c = instru.IFU(
    fov=instru.FOV(6.6/3600, 7.7/3600, origin=instru.Coord(0, 0), angle= - data_dict["rotation"]['4c']),
    det_pix_size=0.273,
    n_slit=12,
    w_blur=spec_blur_4c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('4c'),
    name="4C",
)

main_pointing = instru.Coord(0,0)

pointings = list()
for chan in list_chan:
    pointing_chan = list()
    for obs in range(4):
        RA = data_dict['target'][chan][obs][0]
        DEC = data_dict['target'][chan][obs][1]
        pointing_chan.append(main_pointing + instru.Coord(RA, DEC))
    pointings.append(instru.CoordList(pointing_chan).pix(step_Angle.degree))


alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + data_dict['target']['2a'][2][0] #np.mean(np.array(list_target_c)[:,0])
beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + data_dict['target']['2a'][2][1] #np.mean(np.array(list_target_c)[:,1])

spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT(sotf=sotf, 
                                                    templates=templates, 
                                                    alpha_axis=alpha_axis, 
                                                    beta_axis=beta_axis, 
                                                    wavelength_axis=wavel_axis, 
                                                    instrs=[ch1a, ch1b, ch1c, ch2a, ch2b, ch2c, ch3a, ch3b, ch3c, ch4a, ch4b, ch4c], 
                                                    step_degree=step_Angle.degree, 
                                                    pointings=pointings)


use_data = spectroModel.real_data_janskySR_to_jansky(ndata)

# with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/ch2c_00002_corrected_filtered.fits') as hdul:
#     corrected_slices = hdul[0].data
#     print("corrected slice shape = ", corrected_slices.shape)

# data_shape = (17, 1300, 24)
# # data_shape = (17, 1124, 24)
# # data_shape = (17, 970, 24)
# corrected_slices = corrected_slices.reshape(data_shape[1], data_shape[0], data_shape[2]).transpose(1,0,2)
# # slices_vizualisation.visualize_projected_slices(corrected_slices)
# # slices_vizualisation.visualize_corrected_slices(data_shape, corrected_slices)


# weighted_img, img = spectroModel.plot_slice(ndata, 8, 100)
# weighted_img[weighted_img<10] = np.nanc
# # img[img<10] = np.nan
# # print("Rotation 1a = ", data_dict["rotation"]['1a'])
# # print("Rotation 1b = ", data_dict["rotation"]['1b'])
# # print("Rotation 1c = ", data_dict["rotation"]['1c'])
# # print("Rotation 2a = ", data_dict["rotation"]['2a'])
# # print("Rotation 2b = ", data_dict["rotation"]['2b'])
# # print("Rotation 2c = ", data_dict["rotation"]['2c'])
# # print("Rotation 3a = ", data_dict["rotation"]['3a'])
# # print("Rotation 3b = ", data_dict["rotation"]['3b'])
# # print("Rotation 3c = ", data_dict["rotation"]['3c'])
# # print("Rotation 4a = ", data_dict["rotation"]['4a'])
# # print("Rotation 4b = ", data_dict["rotation"]['4b'])
# # print("Rotation 4c = ", data_dict["rotation"]['4c'])


# plt.figure()
# plt.imshow(np.rot90(np.fliplr(img), -1))
# plt.colorbar()
# plt.figure()
# plt.imshow(weighted_img)
# # plt.imshow(np.rot90(np.fliplr(weighted_img), -1))
# plt.colorbar()
# # plt.clim(vmin=2000)
# plt.figure()
# for i in range(templates.shape[0]):
#     plt.plot(templates[i])
# cube_vizualisation.plot_cube(spsf, wavel_axis)
# plt.show()

# raise RuntimeError("!!!")


# print(dottest(spectroModel, num=10, echo=True))



# cube = spectroModel.adjoint(data)
# cube_vizualisation.plot_cube(cube, spectroModel.wavelength_axis)


# cube_list, wave_list = spectroModel.test_vizual_projection(data)
# cube_vizualisation.plot_concatenated_cubes(cube_list, wave_list)
# spectroModel.project_FOV()

from astropy.wcs import WCS
from astropy.table import Table

# with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/ChannelCube_ch1-long_s3d.fits') as hdul:
#       hdr = hdul[1].header
#       RA = hdr['CRVAL1']
#       DEC = hdr['CRVAL2']
# origin_alpha_axis = np.arange(imshape[0]) * step_Angle.degree
# origin_beta_axis = np.arange(imshape[1]) * step_Angle.degree
# origin_alpha_axis -= np.mean(origin_alpha_axis)
# origin_beta_axis -= np.mean(origin_beta_axis)
# alpha_axis = origin_alpha_axis + RA
# beta_axis = origin_beta_axis + DEC



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




# print("Max Wavel = ", len(wavel_axis))
"""
Reconstruction method
"""
hyperParameter = 5e3
method = "lcg"
niter = 50
value_init = 0

# Create result directory
result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_{len(pointings[0])}_Temp_{templates.shape[0]}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}/'
path = pathlib.Path(result_path+result_dir)
path.mkdir(parents=True, exist_ok=True)

quadCrit_fusion = fusion_CT.QuadCriterion_MRS(mu_spectro=1, 
                                                    y_spectro=np.copy(ndata), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)


y_cube = spectroModel.mapsToCube(res_fusion.x)
flipped_cube = np.array([np.fliplr(y_cube[i]) for i in range(y_cube.shape[0])])

# Save results
print(f"Results save in {path}")
np.save(path / 'res_x.npy', res_fusion.x)
np.save(path / 'res_cube.npy', y_cube)
np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)


# cube_vizualisation.plot_cube(np.rot90(flipped_cube, -1, axes=(1,2)), wavel_axis)
