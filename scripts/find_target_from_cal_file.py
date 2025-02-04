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
from surfh.Vizualisation import slices_vizualisation
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

def extract_name_information(fits_path):
    splitted_filename = fits_path.split('_')
    return splitted_filename[0], splitted_filename[1], splitted_filename[3]


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

    fits_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/ch1b_ch2b_0210j_00001_mirifushort_cal.fits'
    save_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Corrected_slices/'
    mode = [0,1] # 0=1st chan; 1=2nd chan; 2=both chan

    first_chan, second_chan, dithering_number = extract_name_information(os.path.basename(fits_path))
    for mod in mode:
        if mod == 0:
            selected_chan = first_chan
        elif mod==1:
            selected_chan = second_chan
        else:
            raise NameError(f"Error in seleced mode : {mod}")
        
        # Load simulation data
        origin_alpha_axis, origin_beta_axis, wavelength_cube = load_simulation_data()

        # Extract target coordinates from FITS file
        targ_ra, targ_dec = fits_toolbox.get_fits_target_coordinates(fits_path)

        # Setup channel model
        ifu, targ_ra, targ_dec = realmiri.get_IFU(fits_path, channel=selected_chan)
        model_channel = setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube)

        # Process FITS data and generate labeled image
        jwst_model = datamodels.open(fits_path)
        mrs_raw_data = jwst_model.data

        binary_grid = np.zeros_like(mrs_raw_data)
        x_pixel_idx = np.arange(mrs_raw_data.shape[0])
        y_pixel_idx = np.arange(mrs_raw_data.shape[1])
        yy, xx = np.meshgrid(x_pixel_idx, y_pixel_idx)

        detector2world = jwst_model.meta.wcs.get_transform('detector', 'world')
        coordinates = detector2world(xx, yy)
        binary_grid[~np.isnan(coordinates[0].T)] = 1

        label_image = distorsion_correction.generate_label_image(binary_grid)

        # Sort labels 
        sorted_labeled_image = distorsion_correction.sort_labels_by_centroid(label_image)


        i = 0
        list_alpha = list()
        list_beta = list()
        for slit in range(len(np.unique(sorted_labeled_image))):
            
            if slit == 0:
                continue
            
            pixel_set = np.where(sorted_labeled_image == slit)
            alpha, beta, lam = detector2world(pixel_set[1], pixel_set[0])
            
            if mod == 0:
                if np.any(lam > np.max(ifu.wavel_axis) +1):
                    console.log(f"[yellow]Skipping slit due to wavelength limits.[/yellow]")
                    continue
            
            if mod == 1:
                if np.any(lam < np.min(ifu.wavel_axis) -1):
                    console.log(f"[yellow]Skipping slit due to wavelength limits.[/yellow]")
                    continue
            
            list_alpha.append(alpha)
            list_beta.append(beta)
        
        print(f'Mean of alpha is {np.mean([np.mean(means) for means in list_alpha])}')
        print(f'Mean of alpha is {np.mean([np.mean(means) for means in list_beta])}')



if __name__ == "__main__":
    main()