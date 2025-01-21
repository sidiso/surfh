from astropy.io import fits
import numpy as np

def corrected_slices_to_fits(corrected_slices, rotation, target_RA, target_DEC, filename, selected_chan):

    PA_V3 = rotation
    TARG_RA = target_RA
    TARG_DEC = target_DEC

    # Create a PrimaryHDU object to store the 2D image data
    hdu = fits.PrimaryHDU(data=corrected_slices)
    
    # Access the FITS header
    header = hdu.header

    # Add metadata to the header
    header['PA_V3'] = PA_V3   # Position Angle (V3) in degrees
    header['TARG_RA'] = TARG_RA   # Target Right Ascension (in degrees)
    header['TARG_DEC'] = TARG_DEC   # Target Declination (in degrees)

    band = selected_chan[-1]
    if band == 'a':
        header['BAND'] = 'SHORT'
    elif band =='b':
        header['BAND'] = 'MEDIUM'
    elif band == 'c':
        header['BAND'] = 'LONG'
    else:
        raise NameError(f'Band name is not correct : {band}')

    # Create an HDUList to hold the primary HDU
    hdul = fits.HDUList([hdu])

    # Write the data and header to a new FITS file
    hdul.writeto(filename, overwrite=True)


def get_fits_target_coordinates(fits_path):
    """
    Extract target coordinates from FITS header.
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[1].header
        targ_ra = hdr['RA_V1']
        targ_dec = hdr['DEC_V1']

    return targ_ra, targ_dec

def get_fits_target_coordinates_corrected_data(fits_path):
    """
    Extract target coordinates from FITS header.
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        targ_ra = hdr['TARG_RA']
        targ_dec = hdr['TARG_DEC']

    return targ_ra, targ_dec


def get_data_from_fits(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    return data