import numpy as np
import os
from rich import print
from rich.progress import track
from rich.console import Console
from astropy.io import fits

from astropy import units as u
from astropy.coordinates import Angle

from jwst import datamodels

from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.DottestModels import MCMO_SigRLSCT_Channel_Model
from surfh.Preprocessing import distorsion_correction
from surfh.Vizualisation import slices_vizualisation
from surfh.ToolsDir import fits_toolbox


raw_slices_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Raw_slices/'
corrected_slices_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Corrected_slices/'
filtered_slices_dir = '/home/nmonnier/Data/JWST/Orion_bar/Fusion/Filtered_slices/'


for file in sorted(os.listdir(raw_slices_dir)):

    if 'ch3a' in file:
        with fits.open(raw_slices_dir + file ) as hdul: 
            hdr = hdul[1].header
            raw_rotation = hdr['PA_V3']

        for filter_file in sorted(os.listdir(filtered_slices_dir)):
            if 'ch3' in filter_file:
                with fits.open(filtered_slices_dir + filter_file , mode='update') as hdul:   
                    hdr = hdul[0].header
                    hdr['PA_V3'] = raw_rotation + 7.5
                    hdul.flush()

    if 'ch4a' in file:
        with fits.open(raw_slices_dir + file ) as hdul: 
            hdr = hdul[1].header
            raw_rotation = hdr['PA_V3']

        for filter_file in sorted(os.listdir(filtered_slices_dir)):
            if 'ch4' in filter_file:
                with fits.open(filtered_slices_dir + filter_file, mode='update' ) as hdul:   
                    hdr = hdul[0].header
                    hdr['PA_V3'] = raw_rotation + 8.3
                    hdul.flush()