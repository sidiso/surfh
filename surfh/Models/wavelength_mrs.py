from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger

from surfh.Models import instru

if "fits_path" not in globals():
    fits_path = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_fits/'

def modify_fits_path(path):
    global fits_path 
    fits_path = path


def mrs_wavel_axis(filename):
    """Load wavelength axis of a detector from FITS file"""
    with fits.open(fits_path+filename) as hdul:
        hdr = hdul[1].header
        return (np.arange(hdr['NAXIS3']) +hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']

    

def get_mrs_wavelength(chan_name):
    if chan_name == '1a':
        return mrs_wavel_axis('ChannelCube_ch_1_short_s3d_02101_00001.fits')
    elif chan_name == '1b':
        return mrs_wavel_axis('ChannelCube_ch_1_medium_s3d_0210j_00001.fits')
    elif chan_name == '1c':
        return mrs_wavel_axis('ChannelCube_ch_1_long_s3d_02111_00001.fits')
    elif chan_name == '2a':
        return mrs_wavel_axis('ChannelCube_ch_2_short_s3d_02101_00001.fits')
    elif chan_name == '2b':
        return mrs_wavel_axis('ChannelCube_ch_2_medium_s3d_0210j_00001.fits')
    elif chan_name == '2c':
        return mrs_wavel_axis('ChannelCube_ch_2_long_s3d_02111_00001.fits')
    elif chan_name == '3a':
        return mrs_wavel_axis('ChannelCube_ch_3_short_s3d_02101_00001.fits')
    elif chan_name == '3b':
        return mrs_wavel_axis('ChannelCube_ch_3_medium_s3d_0210j_00001.fits')
    elif chan_name == '3c':
        return mrs_wavel_axis('ChannelCube_ch_3_long_s3d_02111_00001.fits')
    else:
        print("ERROR Reading wavelength")