import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import wcs
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
from surfh.Models import realmiri

import pathlib
from pathlib import Path
import scipy

"""
Convert Real data (from Fits file) into numpy cubes or slices
in order to use them with Spectro or SpectroLMM.
"""

# Fits and numpy directories

main_directory = '/home/nmonnier/Data/JWST/Orion_bar/'

fits_directory = main_directory + 'Single_Selected_fits_14062024'
numpy_directory = main_directory + 'Single_numpy_Selected_fits_14062024'
numpy_slices_directory = main_directory + 'Single_slice_numpy_Selected_fits_14062024'
mask_directory = main_directory + 'Single_mask'

fits_name = 'ChannelCube_ch_2_short_s3d_0210f_00001.fits'


"""
Create Model and simulation
"""
origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(4, 0) # subsampling to reduce dim of maps

maps_shape = (maps.shape[0], maps.shape[1], maps.shape[2])
indexes = np.where((wavel_axis>wavelength_mrs.get_mrs_wavelength('1a')[0]) & (wavel_axis<wavelength_mrs.get_mrs_wavelength('2c')[-1]))[0]
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
print(origin_alpha_axis)

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

"""
Process Metadata for all Fits in directory
"""
main_pointing = instru.Coord(0, 0) # Set the main pointing from the instrument FoV
pointings = instru.CoordList([main_pointing])#.pix(step_Angle.degree) # We do not use dithering for first tests
channels = []


split_file  = fits_name.split('_')

ifu, _, _ = realmiri.get_IFU(fits_directory + '/' + fits_name)
channels.append(ifu)

"""
Check If Fits file is already converted to numpy (faster for now)
"""
filename = Path(fits_name).stem
numpy_file = Path(numpy_directory + '/' + filename + '.npy')
numpy_slices_file = Path(numpy_slices_directory + '/' + filename + '.npy')
mask_file = Path(mask_directory + '/' + filename + '.npy')

print("------------------")
print(numpy_file)
if numpy_file.is_file():
    interpolated_cube = np.load(numpy_file)
    print("Numpy file Already computed, load it") 
else:
    print("-------------------")
    """
    Interpolation of real data to oversampled cube
    """
    hdul = fits.open(fits_directory + '/' + fits_name)


    im = hdul[1].data

    # Uncomment to get strongh masking
    # im[:,0:4, :] = np.NaN
    # im[:,-4:-1, :] = np.NaN
    # im[:, :, 0:4] = np.NaN
    # im[:, :, -4:-1] = np.NaN

    # Get wcs metadata from fits header
    w = wcs.WCS(hdul[1].header)

    # Get meshgrid and idx regarding wcs information 
    zz,xx,yy = np.indices(im.shape)
    a, b, c = w.wcs_pix2world(xx, yy, zz, 1)

    wavelength_axis = c[:,0,0]
    ra_map = a[0,:,:]
    dec_map = b[0,:,:]

    ra_map  = ra_map - np.mean(ra_map)
    dec_map = dec_map - np.mean(dec_map)

    of = 3 # Oversampling factor to interpolate fits cube onto thiner pixels
    of_ra_map = np.zeros(((ra_map.shape[0]-2)*of, (ra_map.shape[1]-2)*of), dtype=ra_map.dtype)
    of_dec_map = np.zeros(((dec_map.shape[0]-2)*of, (dec_map.shape[1]-2)*of), dtype=dec_map.dtype)

    # Oversample im with 0 padding, Then dupplicate value to 0 values
    of_im = np.zeros((im.shape[0], im.shape[1]*of, im.shape[2]*of), dtype=im.dtype)
    of_im[:, ::of, ::of] = im
    n_of_im = np.zeros((of_im.shape[0], (of_im.shape[1]-2*of)*of, (of_im.shape[2]-2*of)*of), dtype=im.dtype)
    n_of_im = of_im[:, of:-of, of:-of]
    of_im = n_of_im

    # Convolve im with kernel to dupplicates values
    kern = np.ones((of,of))
    kern = np.pad(kern, (of-1,0))
    for l in range(of_im.shape[0]):
        of_im[l] = scipy.signal.convolve2d(of_im[l], kern, 'same')


    # Get alpha and beta resolution for oversampled pixel 
    ra_step_y = ra_map[5,6] - ra_map[5,5]
    of_ra_step_y = ra_step_y/of
    ra_step_x = ra_map[6,5] - ra_map[5,5]
    of_ra_step_x = ra_step_x/of

    dec_step_y = dec_map[5,6] - dec_map[5,5]
    of_dec_step_y = dec_step_y/of
    dec_step_x = dec_map[6,5] - dec_map[5,5]
    of_dec_step_x = dec_step_x/of


    for i in range(ra_map.shape[0]-2):
        for j in range(ra_map.shape[1]-2): 
            of_ra_map[i*of, j*of] = ra_map[i+1,j+1]

    for i in range(dec_map.shape[0]-2):
        for j in range(dec_map.shape[1]-2):
            of_dec_map[i*of, j*of] = dec_map[i+1,j+1]


    ### Interpolate unknow values
    for i in range(of_ra_map.shape[0]):
        for j in range(of_ra_map.shape[1]):
            if i%of !=0:
                of_ra_map[i, j] = of_ra_map[i-(i%of), j] + of_ra_step_x*(i%of)
            else:
                if j%of != 0:
                    of_ra_map[i, j] = of_ra_map[i, j-(j%of)] + of_ra_step_y*(j%of)

    for i in range(of_dec_map.shape[0]):
        for j in range(of_dec_map.shape[1]):
            if i%of !=0:
                of_dec_map[i, j] = of_dec_map[i-(i%of), j] + of_dec_step_x*(i%of)
            else:
                if j%of != 0:
                    of_dec_map[i, j] = of_dec_map[i, j-(j%of)] + of_dec_step_y*(j%of)



    # 2D Interpolation for each wavelength 
    #of_im = of_im[:-1]
    from scipy.interpolate import griddata
    from scipy.interpolate import NearestNDInterpolator
    interpolated_cube = np.zeros((of_im.shape[0],maps_shape[1],maps_shape[2]))
    for wave in range(of_im.shape[0]):
        # print("Wave %d"%wave)
        TwoD_of_im = of_im[wave].ravel()
        points = np.vstack(
            [
                of_ra_map.ravel(),
                of_dec_map.ravel()         
            ]
            ).T

        alpha_axis = origin_alpha_axis# + hdul[1].header['CRVAL1']
        beta_axis = origin_beta_axis# + hdul[1].header['CRVAL2']
        xi = np.vstack(
            [
                np.tile(alpha_axis.reshape((1, -1)), (len(beta_axis), 1)).ravel(), 
                np.tile(beta_axis.reshape((-1, 1)), (1, len(alpha_axis))).ravel()
            ]
            ).T

        
        interpolated_cube[wave] = griddata(points, 
                                            TwoD_of_im, 
                                            xi, 
                                            method='linear').reshape(maps_shape[1],maps_shape[2])
        data = interpolated_cube[wave]
        mask = np.where(~np.isnan(data))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        filled_data = interp(*np.indices(data.shape))
        interpolated_cube[wave] = filled_data

    # interpolated_cube[np.where(np.isnan(interpolated_cube))] = 0

    np.save(numpy_file, interpolated_cube)



# print(origin_alpha_axis)
# origin_alpha_axis += channels[0].fov.origin.alpha
# print(origin_alpha_axis)
# origin_beta_axis += channels[0].fov.origin.beta


"""
Define instrumental model from previous metadata
"""
spectro = MCMO_SigRLSCT_Model.spectroSigRLSCT_NN(sotf, 
                                              templates, 
                                              origin_alpha_axis, 
                                              origin_beta_axis, 
                                              wavel_axis, 
                                              channels,#, ch3a, ch3b, ch3c], 
                                              step_Angle.degree, 
                                              pointings)

slices = spectro.channels[0].realData_cubeToSlice(interpolated_cube)
slices[np.where(np.isnan(slices))] = 0
np.save(numpy_slices_file, slices)
slices[np.where(np.isnan(slices))] = 0
ncube = spectro.channels[0].realData_sliceToCube(slices, interpolated_cube.shape)

# mask = utils.make_mask_FoV(ncube)
# np.save(mask_file, mask)
# plt.imshow(mask)
# plt.show()
interpolated_cube[np.isnan(interpolated_cube)] = 0

plt.figure()
plt.imshow(interpolated_cube[50])
plt.colorbar()
plt.figure()
plt.imshow(ncube[50])
plt.colorbar()
plt.show() 