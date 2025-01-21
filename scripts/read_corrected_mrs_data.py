import numpy as np
import os
from astropy.io import fits

from astropy import units as u
from astropy.coordinates import Angle

from surfh.Simulation import simulation_data
from surfh.Models import realmiri, instru
from surfh.DottestModels import MCMO_SigRLSCT_Channel_Model
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.ToolsDir import fits_toolbox


def extract_name_information(fits_path):
    splitted_filename = fits_path.split('_')
    return splitted_filename[0], splitted_filename[1]


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

    data_corrected_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Corrected_slices/ch2b_00001_corrected.fits'
    data_corrected_path = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/ch1c_00001_corrected_filtered.fits'
    chan_band, dithering_number = extract_name_information(os.path.basename(data_corrected_path))
    print(chan_band)

    # Load simulation data
    origin_alpha_axis, origin_beta_axis, wavelength_cube = simulation_data.load_simulation_data()

    # Extract target coordinates from FITS file
    targ_ra, targ_dec = fits_toolbox.get_fits_target_coordinates_corrected_data(data_corrected_path)

    # Setup channel model
    ifu, targ_ra, targ_dec = realmiri.get_IFU_from_corrected_data(data_corrected_path, channel=chan_band)
    model_channel = setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ifu, wavelength_cube)
    print(f'Oshape Channel is {model_channel.oshape}')
    print(f"Wavelength min = {np.min(ifu.wavel_axis)}, Wavelength max = {np.max(ifu.wavel_axis)}")
    # Read data from fits
    corrected_data = fits_toolbox.get_data_from_fits(data_corrected_path)

    # Reshape data to fit model   
    data_shape = model_channel.oshape[1:]
    print(f'Data shape is {model_channel.oshape}')

    data = corrected_data.reshape(data_shape[1], data_shape[0], data_shape[2])
    data = data.transpose(1,0,2)


    cube = model_channel.sliceToCube(data)
    print("Cube shape is ", cube.shape)

    cube_vizualisation.plot_cube(cube, wavelength_cube)

if __name__ == "__main__":
    main()