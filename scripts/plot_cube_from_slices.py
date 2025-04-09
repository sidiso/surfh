import numpy as np
import os
from rich import print
from rich.progress import track
from rich.console import Console
from astropy.io import fits

from astropy import units as u
from astropy.coordinates import Angle

import matplotlib.pyplot as plt

from jwst import datamodels

from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.Models import spectroModelChannel
from surfh.Preprocessing import distorsion_correction
from surfh.ToolsDir import fits_toolbox
from surfh.Vizualisation import slices_vizualisation
def load_simulation_data(npix=501):
    """
    Load simulation data and the wavelength information.
    """
    def orion():
        """Rerturn maps, templates, spatial step and wavelength"""
        path_cube_orion='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
        spectrums = fits.open(path_cube_orion + "spectra_mir_orion.fits")[1].data
        wavel_axis = spectrums.wavelength

        return (
            0.025,
            wavel_axis,
        )
    
    step, wavel_axis = orion()
    step_Angle = Angle(step, u.arcsec)
    tpl_ss = 3
    wavel_axis = wavel_axis[::tpl_ss]

    origin_alpha_axis = np.arange(npix) * step_Angle.degree
    origin_beta_axis = np.arange(npix) * step_Angle.degree
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)

    return origin_alpha_axis, origin_beta_axis, wavel_axis



def setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube):
    """
    Set up channel model with super resolution and global wavelength.
    """
    step = 0.025
    step_angle = Angle(step, u.arcsec).degree

    super_resolution_factor = instru.get_srf(
        [ifu.det_pix_size],
        step_angle * 3600,  # Conversion in arcsec
    )

    alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + targ_ra
    beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + targ_dec
    pointings = instru.CoordList([instru.Coord(0, 0)]).pix(step_angle)


    channel = spectroModelChannel.Channel(
        ifu,
        alpha_axis,
        beta_axis,
        wavelength_cube,
        super_resolution_factor[0],
        pointings,
        step_angle
    )
    return channel


ch2a_shape = (17, 970, 24)
ch2b_shape = (17, 1124, 24)
ch2c_shape = (17, 1300, 24)

save_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Observation_2/Fusion/Filtered_slices/'
for file in sorted(os.listdir(save_corrected_dir)):
    if 'ch2a' in file:
        ifu, targ_ra, targ_dec = realmiri.get_IFU(save_corrected_dir + file, chan_name='ch2')
        
        # Load simulation data
        Nx = 501
        origin_alpha_axis, origin_beta_axis, wavelength_cube = load_simulation_data(npix=Nx)

        model_channel = setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube)

        with fits.open(os.path.join(save_corrected_dir, file)) as hdul:
            data = hdul[0].data
            
            ndata = data.reshape(ch2a_shape[1], ch2a_shape[0], ch2a_shape[2])
            ndata = ndata.transpose(1, 0, 2)


            cube = model_channel.realData_sliceToCube(ndata, (ndata.shape[1],501,501))
            plt.imshow(cube[100])
            plt.show()
