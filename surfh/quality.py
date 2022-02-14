#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:09:58 2020

@author: abirizk
"""
import numpy as np
import skimage.measure as measure
#.compare_ssim as cssim
from spectral import spectral_angles

def calc_MSE(object, reconst):
    MSE =  np.mean(
         (object.ravel()
         - reconst.ravel())
         ** 2)
                
    return MSE


def calc_PSNR(vref, vcmp, rng=None):
    """
    Compute Peak Signal to Noise Ratio (PSNR) of two images. The PSNR
    calculation defaults to using the less common definition in terms
    of the actual range (i.e. max minus min) of the reference signal
    instead of the maximum possible range for the data type
    (i.e. :math:`2^b-1` for a :math:`b` bit representation).

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rng : None or int, optional (default None)
      Signal range, either the value to use (e.g. 255 for 8 bit samples) or
      None, in which case the actual range of the reference signal is used

    Returns
    -------
    x : float
      PSNR of `vcmp` with respect to `vref`
    """

    if rng is None:
        rng = vref.max() - vref.min()
    dv = (rng + 0.0) 
    with np.errstate(divide="ignore"):
        rt =  np.where(
                     calc_MSE(vref, vcmp) == 0, float('inf'), dv / np.sqrt(calc_MSE(vref, vcmp)))
          
    return 20.0 * np.log10(rt)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))
#----------------------------------------------------------------------------------------------------------------------------------

def calc_SAM(vref,vcomp):
    
    SAM =  np.where(
                    np.sqrt(np.sum(vref**2)) * np.sqrt(np.sum(vcomp**2)) == 0, 0, \
                    np.arccos(np.sum(vref * vcomp)/(np.sqrt(np.sum(vref**2)) * np.sqrt(np.sum(vcomp**2)))))
    return SAM

def SAM(vref, vcomp):
    return spectral_angles(vref, vcomp)

def relative_error(input, output):
    Error = np.sum(np.abs(
    
            input.ravel()
            - output.ravel()
            )**2
            ) / np.sum(np.abs(input.ravel() )**2)
    return Error * 100

def calc_SSIM(vref, vcomp):    
    
    error_ssim = measure.compare_ssim(vref, vcomp,full=True)
    return error_ssim
#-----------------------------------------------------------------------------------------------------------------------------------------------
def SNR(data, data_wo_noise):
    
    data_flat = []
    data_wonoise_flat = []
    for i, i_wonoise in zip(data, data_wo_noise):
        data_flat.extend(i.flatten())
        data_wonoise_flat.extend(i_wonoise.flatten())
    data_flat = np.asarray(data_flat)
    data_wonoise_flat = np.asarray(data_wonoise_flat)
    SNR = np.sum(np.asarray(data_flat ** 2)) / np.sum((data_flat - data_wonoise_flat) ** 2)
    SNR =  np.nan_to_num(SNR, copy=True)
    SNR_dB = 10 * np.log10(SNR)
    return SNR_dB
