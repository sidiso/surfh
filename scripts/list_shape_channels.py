import numpy as np
import os
from rich import print
from rich.progress import track
from rich.console import Console
from astropy.io import fits

from astropy import units as u
from astropy.coordinates import Angle

from jwst import datamodels

from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.DottestModels import MCMO_SigRLSCT_Channel_Model
from surfh.Preprocessing import distorsion_correction
from surfh.ToolsDir import fits_toolbox

console = Console()

def load_simulation_data():
    """
    Load simulation data and the wavelength information.
    """
    console.log("[bold cyan]Loading simulation data...[/bold cyan]")
    origin_alpha_axis, origin_beta_axis, wavelength_cube, spsf, maps, templates = simulation_data.get_simulation_data(4, 0, '/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/')
    console.log("[bold cyan]Loading wavelength data...[/bold cyan]")
    
    console.log("[green]Simulation data and wavelength loaded successfully![/green]")
    return origin_alpha_axis, origin_beta_axis, wavelength_cube

def setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube):
    """
    Set up channel model with super resolution and global wavelength.
    """
    step = 0.025
    step_angle = Angle(step, u.arcsec).degree
    ch1c_pix = ifu.pix(step_angle)

    super_resolution_factor = instru.get_srf(
        [ifu.det_pix_size],
        step_angle * 3600,  # Conversion in arcsec
    )

    alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + targ_ra
    beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + targ_dec
    pointings = instru.CoordList([instru.Coord(targ_ra, targ_dec)]).pix(step_angle)


    channel = MCMO_SigRLSCT_Channel_Model.Channel(
        ifu,
        alpha_axis,
        beta_axis,
        wavelength_cube,
        super_resolution_factor[0],
        pointings,
        step_angle
    )
    return channel


def main():

    fits_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/ch1c_ch2c_02111_00002_mirifushort_cal.fits'
    save_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Corrected_slices/'
    mode = [0,1] # 0=1st chan; 1=2nd chan; 2=both chan

                
    # Load simulation data
    origin_alpha_axis, origin_beta_axis, wavelength_cube = load_simulation_data()

    # Extract target coordinates from FITS file
    targ_ra, targ_dec = fits_toolbox.get_fits_target_coordinates(fits_path)

    list_chan = ['ch1a', 'ch1b', 'ch1c' , 'ch2a' , 'ch2b' , 'ch2c']
    for chan in list_chan:
        # Setup channel model
        ifu, targ_ra, targ_dec = realmiri.get_IFU(fits_path, channel=chan)
        model_channel = setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube)

        print(f'For Chan {chan}, Oshape Channel is {model_channel.oshape}')

if __name__ == "__main__":
    main()