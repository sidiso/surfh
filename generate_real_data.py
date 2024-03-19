import matplotlib.pyplot as plt
import numpy as np
import time
import os
import udft
from pathlib import Path

from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import Angle

import scipy
from scipy.signal import convolve2d as conv2

from surfh import instru, models
from surfh import utils
from surfh import realmiri
from surfh import cython_2D_interpolation


"""
Save Real data (from Fits file) into numpy cubes or slices
in order to use them with Spectro or SpectroLMM.

"""

# Fits and numpy directories
fits_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_fits'
numpy_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_numpy'




def orion():
    """Rerturn maps, templates, spatial step and wavelength"""
    maps = fits.open("./cube_orion/abundances_orion.fits")[0].data

    h2_map = maps[0]
    if_map = maps[1]
    df_map = maps[2]
    mc_map = maps[3]

    spectrums = fits.open("./cube_orion/spectra_mir_orion.fits")[1].data
    wavel_axis = spectrums.wavelength

    h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
    if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
    df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
    mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

    return (
        np.asarray((h2_map, if_map, df_map, mc_map)),
        np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
        0.025,
        wavel_axis,
    )


maps, tpl, step, wavel_axis = orion()
spatial_subsampling = 4
impulse_response = np.ones((spatial_subsampling, spatial_subsampling)) / spatial_subsampling ** 2
maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in maps])
step_Angle = Angle(step, u.arcsec)

tpl_ss = 3
wavel_axis = wavel_axis[::tpl_ss]
spsf = utils.gaussian_psf(wavel_axis, step_Angle.degree)

if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps.shape[1:])


"""
Set Cube coordinate.
"""
margin=100
maps_shape = (maps.shape[0], maps.shape[1]+margin*2, maps.shape[2]+margin*2)
step_Angle = Angle(step, u.arcsec)
origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)


"""
Process Metadata for all Fits in directory
"""
main_pointing = instru.Coord(0, 0) # Set the main pointing from the instrument FoV
pointings = instru.CoordList([main_pointing])#.pix(step_Angle.degree) # We do not use dithering for first tests
channels = []
for file in os.listdir(fits_directory):
    split_file  = file.split('_')
    
    channels.append(realmiri.get_IFU(fits_directory + '/' + file))

    """
    Check If Fits file is already converted to numpy (faster for now)
    """
    filename = Path(file).stem
    numpy_file = Path(numpy_directory + '/' + filename + '.npy')
    print("------------------")
    print(numpy_file)
    if numpy_file.is_file():
        print("Hey") 
    else:
        print("-------------------")
        """
        Interpolation of real data to oversampled cube
        """
        hdul = fits.open(fits_directory + '/' + file)


        im = hdul[1].data

        # Get wcs metadata from fits header
        w = wcs.WCS(hdul[1].header)

        # Get meshgrid and idx regarding wcs information 
        zz,xx,yy = np.indices(im.shape)
        a, b, c = w.wcs_pix2world(xx, yy, zz, 1)

        wavelength_axis = c[:,0,0]
        ra_map = a[0,:,:]
        dec_map = b[0,:,:]

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

            alpha_axis = origin_alpha_axis + hdul[1].header['CRVAL1']
            beta_axis = origin_beta_axis + hdul[1].header['CRVAL2']
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

        interpolated_cube[np.where(np.isnan(interpolated_cube))] = 0

        np.save(numpy_file, interpolated_cube)



    # fast_insterpolated_cube = cython_2D_interpolation.interpn( (alpha_axis, beta_axis), 
    #                                           np.float64(of_im.newbyteorder('<')),#.byteswap, 
    #                                           xi, 
    #                                           of_im.shape[0],
    #                                           bounds_error=False,
    #                                           fill_value=0).reshape(of_im.shape[0], maps_shape[1],maps_shape[2])




origin_alpha_axis += channels[0].fov.origin.alpha
origin_beta_axis += channels[0].fov.origin.beta


"""
Define instrumental model from previous metadata
"""
""" spectro = models.Spectro(
    channels, # List of channels and bands 
    origin_alpha_axis, # Alpha Coordinates of the cube
    origin_beta_axis, # Beta Coordinates of the cube
    wavel_axis, # Wavelength axis of the cube
    sotf, # Optical PSF
    pointings, # List of pointing (mainly used for dithering)
    verbose=True,
    serial=False,
)

cube = np.load('cube.npy')
slices = spectro.channels[0].realData_cubeToSlice(cube)
ncube = spectro.channels[0].realData_sliceToCube(slices, cube.shape)

plt.figure()
plt.imshow(cube[50])
plt.colorbar()
plt.figure()
plt.imshow(ncube[50])
plt.colorbar()
plt.show() """