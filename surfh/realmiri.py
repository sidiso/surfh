#!/usr/bin/env python3


from importlib import resources

import numpy as np
from astropy.io import fits
from loguru import logger

from . import instru

ARCSEC_TO_DEGREE = 3600

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

"""
Return Instrumental IFU regarding metadata of the fits file.
"""
def get_IFU(filename):
    hdul = fits.open(filename)
    hdr = hdul[0].header

    channel = int(hdr['CHANNEL'])

    if channel == 1:
        slices = 21
        pix_size = 0.196/ARCSEC_TO_DEGREE
        alpha_width = 3.2/ARCSEC_TO_DEGREE
        beta_width = 3.7/ARCSEC_TO_DEGREE
        rotation = 8.4
    elif channel == 2:
        slices = 17
        pix_size = 0.196/ARCSEC_TO_DEGREE
        alpha_width = 4.0/ARCSEC_TO_DEGREE
        beta_width = 4.8/ARCSEC_TO_DEGREE
        rotation = 8.2
    elif channel == 3:
        slices = 16
        pix_size = 0.245/ARCSEC_TO_DEGREE
        alpha_width = 5.5/ARCSEC_TO_DEGREE
        beta_width = 6.2/ARCSEC_TO_DEGREE
        rotation = 7.5
    else:
        slices = 12
        pix_size = 0.273/ARCSEC_TO_DEGREE
        alpha_width = 6.9/ARCSEC_TO_DEGREE
        beta_width = 7.9/ARCSEC_TO_DEGREE
        rotation = 8.3

    if hdr['BAND'] == 'SHORT':
        band = 0
    elif hdr['BAND'] == 'MEDIUM':
        band = 1
    else:
        band = 2 

    print((channel-1)*3 + band)
    spec_blur = instru.SpectralBlur(res[(channel-1)*3 + band])
    targ_ra  = hdr['TARG_RA']
    targ_dec = hdr['TARG_DEC']

    hdr = hdul[1].header
    wavel = (np.arange(hdr['NAXIS3']) +hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']

    if (str(channel) + chr(65 + band)) not in pce:
        pce[str(channel) + chr(65 + band)] = np.random.rand(wavel.size)/10 + 0.5

    hdul.close()
    return instru.IFU(
                        instru.FOV(alpha_width, beta_width, origin=instru.Coord(targ_ra, targ_dec), angle=rotation),
                        pix_size,
                        slices,
                        spec_blur,
                        pce[str(channel) + chr(65 + band)],
                        wavel,
                        str(channel) + chr(65 + band),
                    )
 