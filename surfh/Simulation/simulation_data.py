import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft



def get_simulation_data():
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
    # Multiply maps to match the mean values of real data
    maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in maps])
    step_Angle = Angle(step, u.arcsec)


    """
    Set Cube coordinate.
    """
    margin=0
    maps_shape = (maps.shape[0], maps.shape[1]+margin*2, maps.shape[2]+margin*2)
    step_Angle = Angle(step, u.arcsec)
    origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
    origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)

    tpl_ss = 3
    impulse_response = np.ones((1, tpl_ss)) / tpl_ss
    tpl = conv2(tpl, impulse_response, "same")[:, ::tpl_ss]
    wavel_axis = wavel_axis[::tpl_ss]

    sim_slice = slice(945, 1256, None) # Slice corresponding to chan 2A

    wavel_axis = wavel_axis[sim_slice]
    tpl = tpl[:,sim_slice]
    spsf = np.load('/home/nmonnier/Data/JWST/Orion_bar/All_bands_psf/psfs_pixscale0.025_fov11.25_date_300123.npy')[sim_slice]
    spsf = spsf[:, (100-0):(351+0), (100-0):(351+0)]
    sotf = udft.ir2fr(spsf, maps_shape[1:])

    return origin_beta_axis, origin_beta_axis, wavel_axis, sotf, maps, tpl


