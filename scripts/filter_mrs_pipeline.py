
import numpy as np
import os
from astropy.io import fits

from sklearn.decomposition import NMF
from scipy import ndimage

import matplotlib.pyplot as plt

from pathlib import Path

from surfh.Models import wavelength_mrs



def main():
    save_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Observation_2/Fusion/'
    save_filter_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Observation_2/Fusion/'

    wavelengh_ch1a = 5.5116
    wavelengh_ch1b = 6.1092
    wavelengh_ch1c = 6.9108

    wavelengh_ch2a = 8.02545
    wavelengh_ch2b = 9.66605

    wavelengh_ch3a = 12.1788
    wavelengh_ch3c = 17.0363

    filename = 'Filter_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits'
    fits_name = Path(filename).stem
    with fits.open(save_corrected_dir + fits_name + '.fits') as hdul:
        # Add metadata to the header
        data = hdul[1].data

    data_filtered = data.copy()

    # if 'ch1a' in filename:
    #     wavel_1a = wavelength_mrs.get_mrs_wavelength('ch1a')
    #     idx_wavelength = np.argmin(np.abs(wavel_1a - wavelengh_ch1a))
    #     mask = np.bitwise_not((wavel_1a < wavel_1a[idx_wavelength + 12]) & (wavel_1a > wavel_1a[idx_wavelength - 12]))
    # elif 'ch1b' in filename:
    #     wavel_1b = wavelength_mrs.get_mrs_wavelength('ch1b')
    #     idx_wavelength = np.argmin(np.abs(wavel_1b - wavelengh_ch1b))
    #     mask = np.bitwise_not((wavel_1b < wavel_1b[idx_wavelength + 12]) & (wavel_1b > wavel_1b[idx_wavelength - 12]))
    # elif 'ch1c' in filename:
    #     wavel_1c = wavelength_mrs.get_mrs_wavelength('ch1c')
    #     idx_wavelength = np.argmin(np.abs(wavel_1c - wavelengh_ch1c))
    #     mask = np.bitwise_not((wavel_1c < wavel_1c[idx_wavelength + 12]) & (wavel_1c > wavel_1c[idx_wavelength - 12]))
    # elif 'ch2a' in filename:
    #     wavel_2a = wavelength_mrs.get_mrs_wavelength('ch2a')
    #     idx_wavelength = np.argmin(np.abs(wavel_2a - wavelengh_ch2a))
    #     mask = np.bitwise_not((wavel_2a < wavel_2a[idx_wavelength + 12]) & (wavel_2a > wavel_2a[idx_wavelength - 12]))
    # elif 'ch2b' in filename:
    #     wavel_2b = wavelength_mrs.get_mrs_wavelength('ch2b')
    #     idx_wavelength = np.argmin(np.abs(wavel_2b - wavelengh_ch2b))
    #     mask = np.bitwise_not((wavel_2b < wavel_2b[idx_wavelength + 12]) & (wavel_2b > wavel_2b[idx_wavelength - 12]))
    # elif 'ch3a' in filename:
    #     wavel_3a = wavelength_mrs.get_mrs_wavelength('ch3a')
    #     idx_wavelength = np.argmin(np.abs(wavel_3a - wavelengh_ch3a))
    #     mask = np.bitwise_not((wavel_3a < wavel_3a[idx_wavelength + 12]) & (wavel_3a > wavel_3a[idx_wavelength - 12]))
    # elif 'ch3c' in filename:
    #     wavel_3c = wavelength_mrs.get_mrs_wavelength('ch3c')
    #     idx_wavelength = np.argmin(np.abs(wavel_3c - wavelengh_ch3c))
    #     mask = np.bitwise_not((wavel_3c < wavel_3c[idx_wavelength + 12]) & (wavel_3c > wavel_3c[idx_wavelength - 12]))
    # else:
    #     mask = np.full(data_filtered.shape[0], True, dtype=bool)


    print(f"Data shape is {data.shape}")
    print(f"While wavelength ch1a is {wavelength_mrs.get_mrs_wavelength('ch1a').shape} ")
    # data_filtered[mask] = ndimage.median_filter(data[mask].copy(), size=11, axes=[0])

    with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Observation_2/Fusion/Filter_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits', mode='update') as hdul1:
        # Change something in hdul.
        hdul1[1].data = ndimage.median_filter(data.copy(), size=11, axes=[0])
        hdul1.flush()  # changes are written back to original.fits
       


if __name__ == "__main__":
    main()