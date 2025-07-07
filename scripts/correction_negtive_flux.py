import os
import numpy as np
from astropy.io import fits

from surfh.ToolsDir import correction
import matplotlib.pyplot as plt

def extract_name_corr_filt(dir):
    keywords = dir.split('_')
    return keywords[0], keywords[1]

dir = '/home/nmonnier/Data/JWST/Point_source/Fusion/'

filtered_slices_dir = dir + 'Filtered_slices/'
print(filtered_slices_dir)

for file in os.listdir(filtered_slices_dir):
    chan, dith = extract_name_corr_filt(file)
    print(f"Chan {chan} : Correct MRS {file}")

    with fits.open(filtered_slices_dir+file, mode='update') as hdul:
        header = hdul[0].header

        nslit  = header['NSLITS']
        nwavel = header['NWAVEL']
        nalpha = header['NALPHA'] 

        data = hdul[0].data

        # Correction negative flux per slit
        for slit in range(nslit):
            data_slit = data[:, slit*nalpha:(slit+1)*nalpha]
            if np.any(data_slit<0):
                print(f"Correction Chan {chan} slit {slit}")
                data_corrected = correction.interpolate_negatives_2d(data_slit)
                data[:, slit*nalpha:(slit+1)*nalpha] = data_corrected

        hdul.flush()
