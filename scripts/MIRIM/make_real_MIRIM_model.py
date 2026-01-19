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


npix = 584
SS = 4

fusion_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'
templates_dir = fusion_dir + 'Templates/'
psf_dir = fusion_dir + 'PSF/'


list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W']
imshape = (npix, npix)

step = 0.1  # arcsec
step_angle = step_angle = Angle(step, u.arcsec).degree

ref_list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
otf = np.load(os.path.join(psf_dir, 'mirim_psfs_pixscale0.11091747765212819_npix_584_2602_slices.npy'))
imshape = (otf.shape[1], otf.shape[2])

ref_wavelength = np.load(templates_dir + 'wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy')
wavelength = np.load(templates_dir + 'wavelength/ch1a_to_ch4b_wavel_axis_with_spectral_line.npy')

print(np.digitize(np.setdiff1d(wavelength,ref_wavelength), ref_wavelength))
sp_indexes = np.digitize(np.setdiff1d(wavelength,ref_wavelength), wavelength)



wavelength = wavelength[::SS]
otf = otf[:len(wavelength)]

templates = np.load(templates_dir + 'NMF/full_scan_ch1a_to_ch4b_12_nmf_components_and_10_spectral_lines.npy')


SS_templates = np.zeros((templates.shape[0], len(wavelength)))
for i in range(templates.shape[0]):
    if i < 12:
        SS_templates[i] = templates[i, ::SS]
    idx = np.where(templates[i] != 0)[0]
    for k in idx:
        k_ss = k // SS
        if k_ss < SS_templates.shape[1]:
            SS_templates[i, k_ss] += templates[i, k]




# Select PSF regarding list_filter
indexes = [i for i, val in enumerate(ref_list_filter) if val in list_filter]

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


np.save(templates_dir + 'NMF/full_scan_ch1a_to_ch4b_12_nmf_components_and_10_spectral_lines_SS4.npy', SS_templates)
# np.save(templates_dir + 'wavelength/ch1a_to_ch4b_wavel_axis_with_spectral_line_SS4.npy', wavelength)


MIRIModel = model_creation.create_miri_model(otf, pce, wavelength, SS_templates, imshape, step, None)

hfreq = MIRIModel.H_freq

np.save(templates_dir + 'mirim_fusion_full_scan_hfreq.npy', hfreq)
