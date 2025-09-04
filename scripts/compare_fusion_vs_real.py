from astropy.wcs import WCS
# from reproject import reproject_interp
from matplotlib.patches import Polygon

import numpy as np
import os
import udft
from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt

from rich import print
from rich.progress import track
from rich.console import Console

from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.Models import spectroModel
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT
from aljabr import LinOp, dottest
from scipy import ndimage
from surfh.Simulation.fusion_CT import QuadCriterion_MRS
import argparse
import click
import itertools

import logging as log
from scipy.ndimage import rotate


# Load Fusion results
path = '/home/nmonnier/Data/JWST/Point_source/Fusion/Results/mmmg_MC_6_MO_4_Temp_6_nit_100_mu_5.00e+04_SD_True/'

wavel = np.load(path + 'wavel.npy')
cube = np.load(path + 'res_cube.npy')
fits_file = path + 'res_cube.fits'

with fits.open(fits_file) as hdul:
    hdr = hdul[0].header
    data_fits = hdul[0].data
    
print(hdr)


img = data_fits[100,:,:]
angle = -hdr['PA_V3']
rotated_array = rotate(img, angle, reshape=False, order=3)
print(img.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, origin='lower')
ax[0].set_title('Before rotation')
ax[1].imshow(rotated_array, origin='lower')
ax[1].set_title('After rotation')
plt.show()