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

    

def get_mrs_wavelength(chan_name, fits_path='/home/nmonnier/Data/JWST/Orion_bar/All_bands_fits/'):
    if chan_name == '1a' or chan_name == 'ch1a':
        return global_variables.wavelength_1a
    elif chan_name == '1b' or chan_name == 'ch1b':
        return global_variables.wavelength_1b
    elif chan_name == '1c' or chan_name == 'ch1c':
        return global_variables.wavelength_1c
    elif chan_name == '2a' or chan_name == 'ch2a':
        return global_variables.wavelength_2a
    elif chan_name == '2b' or chan_name == 'ch2b':
        return global_variables.wavelength_2b
    elif chan_name == '2c' or chan_name == 'ch2c':
        return global_variables.wavelength_2c
    else:
        raise ValueError(f"Error Reading wavelength, {chan_name} is not a correct input.")
        