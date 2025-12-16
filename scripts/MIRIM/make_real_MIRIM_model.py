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


npix = 654
SS = 4

fusion_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'
templates_dir = fusion_dir + 'Templates/'
psf_dir = fusion_dir + 'PSF/'


list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W']
imshape = (npix, npix)

step = 0.1  # arcsec
step_angle = step_angle = Angle(step, u.arcsec).degree

ref_list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
otf = np.load(os.path.join(psf_dir, 'mirim_psfs_pixscale0.11091747765212819_npix_654_2602_slices.npy'))
imshape = (otf.shape[1], otf.shape[2])

wavelength = np.load(templates_dir + 'wavelength/ch1a_to_ch4b_wavel_axis_with_spectral_line.npy')
wavelength = wavelength[::SS]
otf = otf[::SS]

# Select PSF regarding list_filter
indexes = [i for i, val in enumerate(ref_list_filter) if val in list_filter]
otf = otf[:len(wavelength)]

# Load PCE -- Don't deal with other multiple wavel now
list_pce = []
for file in sorted(os.listdir(templates_dir + 'MIRIM/')) :
    print(f'Load PCE file for from file {file} ')
    list_pce.append(np.load(templates_dir+'MIRIM/'+file)[0])
pce = np.array(list_pce)
pce = pce[:, ::SS]
pce = pce[indexes, :len(wavelength)]
# Try to load H_freq if exists 

print("pce.shape = ", pce.shape)

templates = np.load(templates_dir + 'NMF/ch1a_to_ch4b_6_nmf_components_and_10_spectral_lines.npy')
templates = templates[:, ::SS]

MIRIModel = model_creation.create_miri_model(otf, pce, wavelength, templates, imshape, step, None)

hfreq = MIRIModel.H_freq

np.save(templates_dir + 'mirim_fusion_hfreq.npy', hfreq)
