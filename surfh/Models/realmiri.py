#!/usr/bin/env python3


from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger
from pathlib import Path

from surfh.Models import instru, wavelength_mrs


ARCSEC_TO_DEGREE = 3600
PCE_PATH = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_pce/'

"""
    Load PCE from reference file.
"""
np.random.seed(150)

global pce
pce = {}

res = []
res.append(np.mean([3320, 3710]))
res.append(np.mean([3190, 3750]))
res.append(np.mean([3100, 3610]))
res.append(np.mean([2990, 3110]))
res.append(np.mean([2750, 3170]))
res.append(np.mean([2860, 3300]))
res.append(np.mean([2530, 2880]))
res.append(np.mean([1790, 2640]))
res.append(np.mean([1980, 2790]))
res.append(np.mean([1460, 1930]))
res.append(np.mean([1680, 1760]))
res.append(np.mean([1630, 1330]))

def get_IFU(filename, channel=None, wavel_from_file=False):
    """
    Return Instrumental IFU regarding metadata of the fits file.
    """
    hdul = fits.open(filename)
    hdr = hdul[0].header
    targ_ra  = hdul[1].header['RA_V1']
    targ_dec = hdul[1].header['DEC_V1']


    rotation_ref = hdul[1].header['PA_V3']

    if channel is None: # We can force a specific channel if mandatory
        channel = int(hdr['CHANNEL'])
    elif '1' in channel:
        channel = 1
    elif '2' in channel:
        channel = 2
    elif '3' in channel:
        channel = 3
    elif '4' in channel:
        channel = 4
    else:
        raise NameError(f"Wrong channel name : {channel}")

    chan_str = ''
    if channel == 1:
        slices = 21
        pix_size = 0.196/ARCSEC_TO_DEGREE
        alpha_width = 3.2/ARCSEC_TO_DEGREE
        beta_width = 3.7/ARCSEC_TO_DEGREE
        rotation = 8.4 + rotation_ref
        chan_str += '1'
    elif channel == 2:
        slices = 17
        pix_size = 0.196/ARCSEC_TO_DEGREE
        alpha_width = 4.0/ARCSEC_TO_DEGREE
        beta_width = 4.8/ARCSEC_TO_DEGREE
        rotation = 8.2 + rotation_ref
        chan_str += '2'
    elif channel == 3:
        slices = 16
        pix_size = 0.245/ARCSEC_TO_DEGREE
        alpha_width = 5.5/ARCSEC_TO_DEGREE
        beta_width = 6.2/ARCSEC_TO_DEGREE
        rotation = 7.5 + rotation_ref
        chan_str += '3'
    else:
        slices = 12
        pix_size = 0.273/ARCSEC_TO_DEGREE
        alpha_width = 6.9/ARCSEC_TO_DEGREE
        beta_width = 7.9/ARCSEC_TO_DEGREE
        rotation = 8.3 + rotation_ref
        chan_str += '4'

    if hdr['BAND'] == 'SHORT':
        band = 0
        chan_str += 'a'
    elif hdr['BAND'] == 'MEDIUM':
        band = 1
        chan_str += 'b'
    else:
        band = 2 
        chan_str += 'c'

    spec_blur = instru.SpectralBlur(res[(channel-1)*3 + band])
    

    hdr = hdul[1].header
    # targ_ra  = hdr['RA_V1'] 
    # targ_dec = hdr['DEC_V1']

    if wavel_from_file:
        wavel = (np.arange(hdr['NAXIS3']) +hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']
    else:
        wavel = wavelength_mrs.get_mrs_wavelength(chan_str)

    if (str(channel) + chr(65 + band)) not in pce:
        pce[str(channel) + chr(65 + band)] = np.random.rand(wavel.size)/10 + 0.5
        #value_pce = np.ones_like(np.load(PCE_PATH + Path(filename).stem + '.pce.npy')) # As PCE is corrected in the pipeline (?), set to 1
        #pce[str(channel) + chr(65 + band)] = value_pce
        

    hdul.close()
    return instru.IFU(
                        # ToChange
                        instru.FOV(alpha_width, beta_width, origin=instru.Coord(targ_ra, targ_dec), angle=rotation),
                        # instru.FOV(alpha_width, beta_width, origin=instru.Coord(0,0), angle=rotation),
                        pix_size*3600,
                        slices,
                        spec_blur,
                        pce[str(channel) + chr(65 + band)],
                        wavel,
                        str(channel) + chr(65 + band),
                    ), targ_ra, targ_dec
 

def get_IFU_from_corrected_data(filename, channel=None, wavel_from_file=False):
    """
    Return Instrumental IFU regarding metadata of the fits file.
    """
    hdul = fits.open(filename)
    hdr = hdul[0].header
    targ_ra  = hdul[0].header['TARG_RA']
    targ_dec = hdul[0].header['TARG_DEC']


    rotation_ref = hdr['PA_V3']

    if channel is None: # We can force a specific channel if mandatory
        channel = int(hdr['CHANNEL'])
    elif '1' in channel:
        channel = 1
    elif '2' in channel:
        channel = 2
    elif '3' in channel:
        channel = 3
    elif '4' in channel:
        channel = 4
    else:
        raise NameError(f"Wrong channel name : {channel}")

    chan_str = ''
    if channel == 1:
        slices = 21
        pix_size = 0.196/ARCSEC_TO_DEGREE
        alpha_width = 3.2/ARCSEC_TO_DEGREE
        beta_width = 3.7/ARCSEC_TO_DEGREE
        rotation = 8.4 + rotation_ref
        chan_str += '1'
    elif channel == 2:
        slices = 17
        pix_size = 0.196/ARCSEC_TO_DEGREE
        alpha_width = 4.0/ARCSEC_TO_DEGREE
        beta_width = 4.8/ARCSEC_TO_DEGREE
        rotation = 8.2 + rotation_ref
        chan_str += '2'
    elif channel == 3:
        slices = 16
        pix_size = 0.245/ARCSEC_TO_DEGREE
        alpha_width = 5.5/ARCSEC_TO_DEGREE
        beta_width = 6.2/ARCSEC_TO_DEGREE
        rotation = 7.5 + rotation_ref
        chan_str += '3'
    else:
        slices = 12
        pix_size = 0.273/ARCSEC_TO_DEGREE
        alpha_width = 6.9/ARCSEC_TO_DEGREE
        beta_width = 7.9/ARCSEC_TO_DEGREE
        rotation = 8.3 + rotation_ref
        chan_str += '4'

    if hdr['BAND'] == 'SHORT':
        band = 0
        chan_str += 'a'
    elif hdr['BAND'] == 'MEDIUM':
        band = 1
        chan_str += 'b'
    else:
        band = 2 
        chan_str += 'c'

    spec_blur = instru.SpectralBlur(res[(channel-1)*3 + band])
    

    wavel = wavelength_mrs.get_mrs_wavelength(chan_str)
    print(f'Wavek shape is {wavel.shape}')

    if (str(channel) + chr(65 + band)) not in pce:
        pce[str(channel) + chr(65 + band)] = np.random.rand(wavel.size)/10 + 0.5
        #value_pce = np.ones_like(np.load(PCE_PATH + Path(filename).stem + '.pce.npy')) # As PCE is corrected in the pipeline (?), set to 1
        #pce[str(channel) + chr(65 + band)] = value_pce
    
    hdul.close()

    return instru.IFU(
                        # ToChange
                        # instru.FOV(alpha_width, beta_width, origin=instru.Coord(targ_ra, targ_dec), angle=rotation),
                        instru.FOV(alpha_width, beta_width, origin=instru.Coord(0,0), angle=rotation),
                        pix_size*3600,
                        slices,
                        spec_blur,
                        pce[str(channel) + chr(65 + band)],
                        wavel,
                        str(channel) + chr(65 + band),
                    ), targ_ra, targ_dec
