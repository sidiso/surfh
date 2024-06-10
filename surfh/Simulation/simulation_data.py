import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft



def get_simulation_data(spatial_subsampling=4, margin=0):
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

    origin_size_axe = 0
    if margin != 0:
        origin_size_axe = maps[0,::spatial_subsampling, ::spatial_subsampling].shape[1]
        spatial_subsampling = spatial_subsampling - 1

    if origin_size_axe + 2*margin > maps.shape[1]:
        raise Exception(f"The margin is too large !!")
        
    spatial_subsampling = spatial_subsampling
    impulse_response = np.ones((spatial_subsampling, spatial_subsampling)) / spatial_subsampling ** 2
    # Multiply maps to match the mean values of real data
    maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in maps])
    step_Angle = Angle(step, u.arcsec)

    if margin != 0:
        new_size_axe = maps.shape[1]

        # Select maps to be the same shape as the origin maps + margin
        idx = maps.shape[1]//2 # Center of the spsf
        N = origin_size_axe + margin*2 # Size of the window
        if N%2:
            stepidx = N//2
        else:
            stepidx = int(N/2) - 1
        start = min(max(idx-stepidx, 0), maps.shape[1]-N)
        #spsf = spsf[:, (100-0):(351+0), (100-0):(351+0)]
        maps = maps[:, start:start+N, start:start+N]

    """
    Set Cube coordinate.
    """
    
    tpl_ss = 3
    impulse_response = np.ones((1, tpl_ss)) / tpl_ss
    tpl = conv2(tpl, impulse_response, "same")[:, ::tpl_ss]
    wavel_axis = wavel_axis[::tpl_ss]

    
    spsf = np.load('/home/nmonnier/Data/JWST/Orion_bar/All_bands_psf/psfs_pixscale0.025_fov11.25_date_300123.npy')#[sim_slice]
    if maps.shape[1] > spsf.shape[1]:
        diff = maps.shape[1] - spsf.shape[1]
        if  diff%2:
            maps = maps[:,slice(diff//2+1, maps.shape[1]-diff//2,None),:]
        else:
            maps = maps[:,slice(diff//2, maps.shape[1]-diff//2,None),:]

    if maps.shape[2] > spsf.shape[2]:
        diff = maps.shape[2] - spsf.shape[2]
        if  diff%2:
            maps = maps[:, :, slice(diff//2+1,maps.shape[2]-diff//2,None)]
        else:
            maps = maps[:, :, slice(diff//2, maps.shape[2]-diff//2,None)]

    # If maps is not the size of the psf (can't be bigger)
    if maps.shape[1] != spsf.shape[1]:
        if maps.shape[1] >= spsf.shape[1]:
            print("1")
            maps_shape = (maps.shape[0], spsf.shape[1], spsf.shape[2])
        else:
            print("2")
            maps
            maps_shape = (maps.shape[0], maps.shape[1], maps.shape[2])
    else:
        print("3")
        maps_shape = (maps.shape[0], maps.shape[1], maps.shape[2])

    step_Angle = Angle(step, u.arcsec)
    origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
    origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)

    # sim_slice = slice(945, 1256, None) # Slice corresponding to chan 2A

    # wavel_axis = wavel_axis[sim_slice]
    # tpl = tpl[:,sim_slice]
    

    # # Select PSF to be the same shape as maps
    # idx = spsf.shape[1]//2 # Center of the spsf
    # N = maps.shape[1] # Size of the window
    # if N%2:
    #     stepidx = N//2
    # else:
    #     stepidx = int(N/2) - 1
    # start = min(max(idx-stepidx, 0), spsf.shape[1]-N)
    # #spsf = spsf[:, (100-0):(351+0), (100-0):(351+0)]
    # spsf = spsf[:, start:start+N, start:start+N]
    # sotf = udft.ir2fr(spsf, maps_shape[1:])

    return origin_beta_axis, origin_beta_axis, wavel_axis, spsf, maps, tpl


