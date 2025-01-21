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


"""
Create Model and simulation
"""


fusion_dir_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/'

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

wavel_axis = np.load(template_dir_path + 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_6_templates_SS4.npy')
templates = np.load(template_dir_path + 'nmf_orion_1ABC_2ABC_3ABC_4ABC_6_templates_SS4.npy')


print("tempaltes shape = ", templates.shape)
if len(templates.shape) == 1:
      templates = templates[np.newaxis,...]

templates = templates/10e3


grating_resolution_1a = np.mean([3320, 3710])
spec_blur_1a = instru.SpectralBlur(grating_resolution_1a)
# Def Channel spec.
ch1a = instru.IFU(
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(3.2/3600, 3.7/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(5.2/3600, 6.2/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(5.2/3600, 6.2/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(5.2/3600, 6.2/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(6.6/3600, 7.7/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(6.6/3600, 7.7/3600, origin=instru.Coord(0, 0), angle= 0),
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
    fov=instru.FOV(6.6/3600, 7.7/3600, origin=instru.Coord(0, 0), angle= 0),
    det_pix_size=0.273,
    n_slit=12,
    w_blur=spec_blur_4c,
    pce=None,
    wavel_axis=wavelength_mrs.get_mrs_wavelength('4c'),
    name="4C",
)


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
list_chan = ['1a', '1b', '1c' , '2a' , '2b' , '2c', '3a', '3b', '3c', '4a', '4b', '4c']
data_dict = load_data(list_chan, save_filter_corrected_dir)


# alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + data_dict['target']['2a'][2][0] #np.mean(np.array(list_target_c)[:,0])
# beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + data_dict['target']['2a'][2][1] #np.mean(np.array(list_target_c)[:,1])

with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/ch1a_ch2a_02101_00003_mirifushort_cal.fits') as hdul:
      hdr = hdul[1].header
      RA = hdr['RA_REF']
      DEC = hdr['DEC_REF']
alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + RA
beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + DEC


main_pointing = instru.Coord(0,0)

pointings = list()
for chan in range(12):
    pointing_chan = list()
    for obs in range(1):
        RA = 0
        DEC = 0
        pointing_chan.append(main_pointing + instru.Coord(RA, DEC))
    pointings.append(instru.CoordList(pointing_chan).pix(step_Angle.degree))

spectroModel = MCMO_SigRLSCT_Model.spectroSigRLSCT(sotf=None, 
                                                    templates=templates, 
                                                    alpha_axis=alpha_axis, 
                                                    beta_axis=beta_axis, 
                                                    wavelength_axis=wavel_axis, 
                                                    instrs=[ch1a, ch1b, ch1c, ch2a, ch2b, ch2c, ch3a, ch3b, ch3c, ch4a, ch4b, ch4c], 
                                                    step_degree=step_Angle.degree, 
                                                    pointings=pointings)

maps = np.load(fusion_dir_path + 'Results/lcg_MC_12_MO_4_Temp_6_nit_200_mu_1.00e+03_SD_True/res_x.npy')
masks = np.load(fusion_dir_path + 'Masks/binary_mask_1ABC_2ABC_3ABC_4ABC.npy')

ch1_idx = np.argmin(np.abs(wavel_axis - 7.55))
ch2_idx = np.argmin(np.abs(wavel_axis - 11.63))
ch3_idx = np.argmin(np.abs(wavel_axis - 17.8))

ch1_slice = slice(0, ch1_idx)
ch2_slice = slice(ch1_idx, ch2_idx)
ch3_slice = slice(ch2_idx, ch3_idx)
ch4_slice = slice(ch3_idx, len(wavel_axis))

y_cube = spectroModel.mapsToCube(maps)
y_cube[ch1_slice] = y_cube[ch1_slice] * masks[0]
y_cube[ch2_slice] = y_cube[ch2_slice] * masks[1]
y_cube[ch3_slice] = y_cube[ch3_slice] * masks[2]
y_cube[ch4_slice] = y_cube[ch4_slice] * masks[3]
print(ch1_slice,ch2_slice,ch3_slice,ch4_slice)
print(wavel_axis)

flipped_cube = np.array([np.fliplr(y_cube[i]) for i in range(y_cube.shape[0])])

hdul = fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits')
data_cube = hdul[1].data
hdr = hdul[1].header
wavel = np.array(hdul[5].data[0])[0,:,0]

from astropy.wcs import WCS

# Open the FITS file and load the full HDU list
with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Fusion/ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits') as hdulist:
    header = hdul[1].header  # Extract the SCI extension header
    CRPIX1 = header["CRPIX1"]
    CRPIX2 = header["CRPIX2"]
    CRVAL1 = header["CRVAL1"]
    CRVAL2 = header["CRVAL2"]
    CDELT1 = header["CDELT1"]
    CDELT2 = header["CDELT2"]
    PC1_1 = -header["PC1_1"]
    PC1_2  = header["PC1_2"]
    PC2_1  = header["PC2_1"]
    PC2_2  = header["PC2_2"]
    NAXIS1 = header["NAXIS1"] 
    NAXIS2 = header["NAXIS2"]

# Create pixel coordinate grid
x = np.arange(1, NAXIS1 + 1)  # Pixel x-coordinates
y = np.arange(1, NAXIS2 + 1)  # Pixel y-coordinates
X, Y = np.meshgrid(x, y)      # Create grid of pixel coordinates

# Compute pixel offsets from reference pixel
delta_x = X - CRPIX1
delta_y = Y - CRPIX2

# Apply the linear transformation
RA_offset = delta_x * CDELT1 * PC1_1 + delta_y * CDELT1 * PC1_2
DEC_offset = delta_x * CDELT2 * PC2_1 + delta_y * CDELT2 * PC2_2

# Compute RA and DEC
RA = CRVAL1 + RA_offset
DEC = CRVAL2 + DEC_offset


# Fonction pour réduire le vecteur A
def reduce_vector(A, coords_A, coords_B):
    reduced_A = []
    for coord in coords_B:
        # Trouver l'indice de la coordonnée la plus proche
        closest_idx = np.argmin(np.abs(coords_A - coord))
        # Ajouter la valeur correspondante
        reduced_A.append(A[closest_idx])
    return np.array(reduced_A)

viz_pipeline_cube = reduce_vector(data_cube, wavel, wavel_axis)
print("viz_pipeline_cube Shape is ", viz_pipeline_cube.shape)

flipped_data_cube = np.array([np.rot90(np.fliplr(viz_pipeline_cube[i]), -1) for i in range(viz_pipeline_cube.shape[0])])




from scipy.interpolate import RegularGridInterpolator


""" 
Verticale Line Fusion
"""
ra1, dec1 = 83.835939, -5.416140
ra2, dec2 = 83.832803, -5.417226

# Create the line
num_points = 1000  # Number of points to sample along the line
t = np.linspace(0, 1, num_points)
ra_line = ra1 + t * (ra2 - ra1)
dec_line = dec1 + t * (dec2 - dec1)

edge_array = np.zeros((y_cube.shape[0], num_points))
# Interpolate values from img along the line
for i in range(y_cube.shape[0]):
    interpolator = RegularGridInterpolator((beta_axis, alpha_axis), flipped_cube[i], bounds_error=False, fill_value=0)
    points = np.array([dec_line, ra_line]).T
    values_on_line = interpolator(points)
    values_on_line = values_on_line/np.max(values_on_line)
    edge_array[i] = values_on_line
print("data cube shape", data_cube.shape)



flipped_data_cube[np.isnan(flipped_data_cube)] = 0
edge_array_real_data = np.zeros((flipped_data_cube.shape[0], num_points))
# Interpolate values from img along the line
for i in range(flipped_data_cube.shape[0]):
    interpolator = RegularGridInterpolator((DEC[:,0], RA[0]), flipped_data_cube[i], bounds_error=False, fill_value=0)
    points = np.array([dec_line, ra_line]).T
    values_on_line = interpolator(points)
    values_on_line = values_on_line/np.max(values_on_line)
    edge_array_real_data[i] = values_on_line



# Plot the image with the line
plt.figure(figsize=(10, 10))
plt.imshow(flipped_cube[1500], extent=[alpha_axis[0], alpha_axis[-1], beta_axis[0], beta_axis[-1]], origin='lower', aspect='auto', cmap='viridis')
plt.plot(ra_line, dec_line, color='red', linewidth=2, label='Line')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('Fusion with Line Overlay')
plt.legend()
plt.grid(False)


# Plot the image with the line
plt.figure(figsize=(10, 10))
plt.imshow(flipped_data_cube[1500], extent=[RA[0,0], RA[0,-1], DEC[0,0], DEC[-1,0]], origin='lower', aspect='auto', cmap='viridis')
plt.plot(ra_line, dec_line, color='red', linewidth=2, label='Line')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('Real data with Line Overlay')
plt.legend()
plt.grid(False)



interpolator = RegularGridInterpolator((beta_axis, alpha_axis), flipped_cube[i], bounds_error=False, fill_value=0)
points = np.array([dec_line, ra_line]).T
values_on_line_fusion = interpolator(points)
values_on_line_fusion = values_on_line_fusion/np.max(values_on_line_fusion)
# Plot the extracted values
plt.figure(figsize=(10, 6))
plt.plot(range(num_points), values_on_line, label='Values along the line')
plt.xlabel('Position along the line')
plt.ylabel('Value')
plt.title('')
plt.legend()
plt.grid()

plt.figure()
plt.imshow(edge_array)
plt.title('Fusion')
plt.colorbar()

plt.figure()
plt.imshow(edge_array_real_data)
plt.title('Real data')
plt.colorbar()


plt.show()
