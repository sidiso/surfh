import numpy as np

import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle


from rich import print
import udft

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Vizualisation import cube_vizualisation


npix = 125

fusion_dir = '/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/'
templates_dir = fusion_dir + 'Templates/'
psf_dir = fusion_dir + 'PSF/'


list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W']
imshape = (npix, npix)

step = 0.1  # arcsec
step_angle = step_angle = Angle(step, u.arcsec).degree

ref_list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
otf = np.load(os.path.join(psf_dir, 'mirim_psfs_pixscale0.1_npix_125.npy'))
imshape = (otf.shape[1], otf.shape[2])

wavelength = np.load(templates_dir + 'wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy')

# Select PSF regarding list_filter
indexes = [i for i, val in enumerate(ref_list_filter) if val in list_filter]
otf = otf[:len(wavelength)]

# Load PCE -- Don't deal with other multiple wavel now
list_pce = []
for file in sorted(os.listdir(templates_dir + 'MIRIM/')) :
    print(f'Load PCE file for from file {file} ')
    list_pce.append(np.load(templates_dir+'MIRIM/'+file)[0])
pce = np.array(list_pce)
pce = pce[indexes, :len(wavelength)]
# Try to load H_freq if exists 

print("pce.shape = ", pce.shape)

templates = np.load(templates_dir + 'simulation_templates.npy')
print("templates.shape = ", templates.shape)

MIRIModel = model_creation.create_miri_model(otf, pce, wavelength, templates, imshape, step, None)

hfreq = MIRIModel.H_freq

np.save(templates_dir + 'simulation_hfreq.npy', hfreq)
