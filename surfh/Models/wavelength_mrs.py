from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger

from surfh.Models import instru
from surfh.Others import global_variables



def mrs_wavel_axis(filename, fits_path):
    """Load wavelength axis of a detector from FITS file"""
    with fits.open(fits_path+filename) as hdul:
        hdr = hdul[1].header
        return (np.arange(hdr['NAXIS3']) +hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']

    

def get_mrs_wavelength(chan_name):
    if chan_name == '1a' or chan_name == 'ch1a' or chan_name == '1A' or chan_name == 'ch1A':
        return global_variables.wavelength_1a
    elif chan_name == '1b' or chan_name == 'ch1b' or chan_name == '1B' or chan_name == 'ch1B':
        return global_variables.wavelength_1b
    elif chan_name == '1c' or chan_name == 'ch1c' or chan_name == '1C' or chan_name == 'ch1C':
        return global_variables.wavelength_1c
    elif chan_name == '2a' or chan_name == 'ch2a' or chan_name == '2A' or chan_name == 'ch2A':
        return global_variables.wavelength_2a
    elif chan_name == '2b' or chan_name == 'ch2b' or chan_name == '2B' or chan_name == 'ch2B':
        return global_variables.wavelength_2b
    elif chan_name == '2c' or chan_name == 'ch2c' or chan_name == '2C' or chan_name == 'ch2C':
        return global_variables.wavelength_2c
    elif chan_name == '3a' or chan_name == 'ch3a' or chan_name == '3A' or chan_name == 'ch3A':
        return global_variables.wavelength_3a
    elif chan_name == '3b' or chan_name == 'ch3b' or chan_name == '3B' or chan_name == 'ch3B':
        return global_variables.wavelength_3b
    elif chan_name == '3c' or chan_name == 'ch3c' or chan_name == '3C' or chan_name == 'ch3C':
        return global_variables.wavelength_3c
    elif chan_name == '4a' or chan_name == 'ch4a' or chan_name == '4A' or chan_name == 'ch4A':
        return global_variables.wavelength_4a
    elif chan_name == '4b' or chan_name == 'ch4b' or chan_name == '4B' or chan_name == 'ch4B':
        return global_variables.wavelength_4b
    elif chan_name == '4c' or chan_name == 'ch4c' or chan_name == '4C' or chan_name == 'ch4C':
        return global_variables.wavelength_4c
    else:
        raise ValueError(f"Error Reading wavelength, {chan_name} is not a correct input.")
        