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

console = Console()

def load_simulation_data(npix=125, step=0.1):
    """
    Load simulation data and the wavelength information.
    """
    def orion():
        """Rerturn maps, templates, spatial step and wavelength"""
        path_cube_orion='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
        spectrums = fits.open(path_cube_orion + "spectra_mir_orion.fits")[1].data
        wavel_axis = spectrums.wavelength

        return wavel_axis
    
    wavel_axis = orion()
    step_Angle = Angle(step, u.arcsec)
    tpl_ss = 3
    wavel_axis = wavel_axis[::tpl_ss]


    console.log("[bold cyan]Loading wavelength data...[/bold cyan]")
    origin_alpha_axis = np.arange(npix) * step_Angle.degree
    origin_beta_axis = np.arange(npix) * step_Angle.degree
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)


    console.log("[green]Simulation data and wavelength loaded successfully![/green]")
    return origin_alpha_axis, origin_beta_axis, wavel_axis

def extract_name_information(fits_path):
    splitted_filename = fits_path.split('_')
    return splitted_filename[0], splitted_filename[1], splitted_filename[3]


def setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube, step):
    """
    Set up channel model with super resolution and global wavelength.
    """
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



def main():

    dir = '/home/nmonnier/Data/JWST/Point_source/Fusion/'
    save_corrected_dir = dir + 'Corrected_slices/'
    mode = [0,1] # 0=1st chan; 1=2nd chan; [0,1]=both chan
    raw_dir = dir + 'Raw_slices/'
    for file in sorted(os.listdir(raw_dir)):
        print("------------------------------------------------------------------------")
        console.log(f"[bold blue]--- Processing file : {file} ---[/bold blue]")
        first_chan, second_chan, dithering_number = extract_name_information(os.path.basename(raw_dir + file))

        # # DEBUG
        # if 'ch1a_ch2a_04102' not in file:
        #     continue

        for mod in mode:
            if mod == 0:
                selected_chan = first_chan
            elif mod==1:
                selected_chan = second_chan
            else:
                raise NameError(f"Error in seleced mode : {mod}")

            console.log(f"[bold blue]--- Chan selected is {selected_chan}! ---[/bold blue]")


            # Load simulation data
            Nx = 125
            step = 0.1 # arcsec
            origin_alpha_axis, origin_beta_axis, wavelength_cube = load_simulation_data(npix=Nx, step=step)

            # Extract target coordinates from FITS file
            targ_ra, targ_dec = fits_toolbox.get_fits_target_coordinates(raw_dir + file)

            # Setup channel model
            ifu, targ_ra, targ_dec = realmiri.get_IFU(raw_dir + file, chan_name=selected_chan)
            model_channel = setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube, step)

            # Process FITS data and generate labeled image
            jwst_model = datamodels.open(raw_dir + file)
            mrs_raw_data = jwst_model.data

            binary_grid = np.zeros_like(mrs_raw_data)
            x_pixel_idx = np.arange(mrs_raw_data.shape[0])
            y_pixel_idx = np.arange(mrs_raw_data.shape[1])
            yy, xx = np.meshgrid(x_pixel_idx, y_pixel_idx)

            detector2world = jwst_model.meta.wcs.get_transform('detector', 'world')
            print(detector2world.inputs)
            coordinates = detector2world(xx, yy)
            binary_grid[~np.isnan(coordinates[0].T)] = 1

            label_image = distorsion_correction.generate_label_image(binary_grid)

            # Sort labels 
            sorted_labeled_image = distorsion_correction.sort_labels_by_centroid(label_image)
            print(model_channel.oshape)
            print(model_channel.oshape[1:])

            # # Debug
            # mean_alpha, mean_beta = distorsion_correction.mrs_slices_debugging(model_channel, sorted_labeled_image, detector2world, mrs_raw_data, ifu.wavel_axis, mod)
            # print(f"For file {file} and channel {selected_chan}, mean alpha is {mean_alpha} and mean beta is {mean_beta}")
            # continue

            corrected_slices = distorsion_correction.mrs_slices_distrorsion_correction(model_channel, 
                                                                                        sorted_labeled_image, 
                                                                                        detector2world, 
                                                                                        mrs_raw_data, 
                                                                                        ifu.wavel_axis,
                                                                                        mod)
            slices_shape = corrected_slices.shape
            console.log("[bold blue]--- Process completed successfully! ---[/bold blue]")

            # # Sort slices in the right order
 
            if 'ch1' in selected_chan:
                sorted_data = np.zeros_like(corrected_slices)
                new_order = [10, 20, 9, 19, 8, 18, 7, 17, 6, 16, 5, 15, 4, 14, 3, 13, 2, 12, 1, 11, 0]
                for i in range(corrected_slices.shape[0]):
                    sorted_data[new_order[i]] = corrected_slices[i]

            elif 'ch2' in selected_chan:
                sorted_data = np.zeros_like(corrected_slices)
                new_order = [0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8]
                for i in range(corrected_slices.shape[0]):
                    sorted_data[new_order[i]] = corrected_slices[i]

            elif 'ch3' in selected_chan:
                sorted_data = np.zeros_like(corrected_slices)
                new_order = [15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0]
                for i in range(corrected_slices.shape[0]):
                    sorted_data[new_order[i]] = corrected_slices[i]

            elif 'ch4' in selected_chan:
                sorted_data = np.zeros_like(corrected_slices)
                new_order = [11, 5, 10, 4, 9, 3, 8, 2, 7, 1, 6, 0]
                for i in range(corrected_slices.shape[0]):
                    sorted_data[new_order[i]] = corrected_slices[i]
            else:
                raise NameError(f'Error The name of the selected chan is wrong : {selected_chan}')


            # cube = model_channel.realData_sliceToCube(sorted_data, (sorted_data.shape[1],501,501))
            # plt.imshow(cube[100])
            # plt.show()

            filename = save_corrected_dir + selected_chan + '_' + dithering_number + '_corrected.fits'
            corrected_slice_fits = sorted_data.transpose(1, 0, 2).reshape(sorted_data.shape[1], sorted_data.shape[2]*sorted_data.shape[0])
            



            fits_toolbox.corrected_slices_to_fits(corrected_slice_fits, ifu.fov.angle, targ_ra, targ_dec, filename, selected_chan, slices_shape)

            # slices_vizualisation.visualize_corrected_slices(corrected_slices.shape, corrected_slices)

if __name__ == "__main__":
    main()