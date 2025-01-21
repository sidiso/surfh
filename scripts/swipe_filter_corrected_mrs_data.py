
import numpy as np
import os
from astropy.io import fits

from sklearn.decomposition import NMF
from scipy import ndimage

import matplotlib.pyplot as plt

from pathlib import Path




def main():
    save_filter_corrected_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/'

    for filename in os.listdir(save_filter_corrected_dir):  # iterates over all the files in 'path'
        fits_name = Path(filename).stem
        if 'ch2' in fits_name:
            with fits.open(save_filter_corrected_dir + fits_name + '.fits') as hdul:
                header = hdul[0].header
                # Add metadata to the header
                PA_V3 = header['PA_V3'] # Position Angle (V3) in degrees
                TARG_RA = header['TARG_RA'] # Target Right Ascension (in degrees)
                TARG_DEC = header['TARG_DEC'] # Target Declination (in degrees)
                band = header['BAND']

                data = hdul[0].data


            blocks = [data[:, i*24:(i+1)*24] for i in range(17)]

            # Inverse l'ordre des blocs
            swapped_blocks = blocks[::-1]

            # Recombine les blocs échangés
            result = np.hstack(swapped_blocks)
            # Create a PrimaryHDU object to store the 2D image data
            hdu = fits.PrimaryHDU(data=result)
            # Access the FITS header
            header = hdu.header

            # Add metadata to the header
            header['PA_V3'] = PA_V3   # Position Angle (V3) in degrees
            header['TARG_RA'] = TARG_RA   # Target Right Ascension (in degrees)
            header['TARG_DEC'] = TARG_DEC   # Target Declination (in degrees)

            header['BAND'] = band

            # Create an HDUList to hold the primary HDU
            hdul = fits.HDUList([hdu])
            # Write the data and header to a new FITS file
            hdul.writeto(save_filter_corrected_dir + fits_name + '.fits', overwrite=True)


if __name__ == "__main__":
    main()