import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

path_cube_orion = '/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'

maps = fits.open(path_cube_orion + "abundances_orion.fits")[0].data
spectrums = fits.open(path_cube_orion + "spectra_mir_orion.fits")[1].data
wavels = spectrums['wavelength']

h2_map = maps[0]
if_map = maps[1]
df_map = maps[2]
mc_map = maps[3]
maps = np.asarray((h2_map, if_map, df_map, mc_map))

h2_spectrum = spectrums["spectrum_h2"]
if_spectrum = spectrums["spectrum_if"]
df_spectrum = spectrums["spectrum_df"]
mc_spectrum = spectrums["spectrum_mc"]
templates = np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum])

templates = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/NMF/full_scan_simplified_ch1b_to_ch4c_12_nmf_components_and_8_spectral_lines.npy')

slice_idx = 500
slice_img = np.zeros(maps.shape[1:])
for i in range(maps.shape[0]):
    slice_img += maps[i]*templates[i,slice_idx]

x_cut = 200
y_cut = 50

# Apply median filter to each spectrum of templates
from scipy.ndimage import median_filter
for i in range(templates.shape[0]):
    templates[i] = median_filter(templates[i], size=7)

print(wavels.shape, templates.shape)
plt.figure(figsize=(12, 6))
plt.xticks(fontsize=16)
plt.plot(wavels[:len(templates[0])], templates[0], label='H2', color='blue', linewidth=2.5)
plt.xlabel('Wavelength (microns)', fontsize=18)
plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/h2_spectrum.png', dpi=800)
plt.figure(figsize=(12, 6))
plt.xticks(fontsize=16)
plt.plot(wavels[:len(templates[1])], templates[1], label='IF', color='green', linewidth=2.5)
plt.xlabel('Wavelength (microns)', fontsize=18)
plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/if_spectrum.png', dpi=800)
plt.figure(figsize=(12, 6))
plt.xticks(fontsize=16)
plt.xlabel('Wavelength (microns)', fontsize=18)
plt.plot(wavels[:len(templates[2])], templates[2], label='DF', color='orange', linewidth=2.5)
plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/df_spectrum.png', dpi=800)
plt.figure(figsize=(12, 6))
plt.xlabel('Wavelength (microns)', fontsize=18)
plt.xticks(fontsize=16)
plt.plot(wavels[:len(templates[3])], templates[3], label='MC', color='red', linewidth=2.5)
plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/mc_spectrum.png', dpi=800)

plt.figure(figsize=(12, 6))
plt.xticks(fontsize=16)
plt.xlabel('Wavelength (microns)', fontsize=18)
plt.plot(wavels[:len(templates[0])], np.mean(templates[:3], axis=0), label='Mean of H2, IF, DF', color='black', linewidth=2.5)
plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/mean_spectrum.png', dpi=800)
plt.show()
# plt.legend()
# plt.figure()
# plt.imshow(slice_img[130:-650, 200:-100])
# plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/lmm.png', dpi=800)

# plt.figure()
# plt.imshow(maps[0,130:-650, 200:-100])
# plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/h2_map.png', dpi=800)

# plt.figure()
# plt.imshow(maps[1,130:-650, 200:-100])
# plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/if_map.png', dpi=800)

# plt.figure()
# plt.imshow(maps[2,130:-650, 200:-100])
# plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/df_map.png', dpi=800)

# plt.figure()
# plt.imshow(maps[3,130:-650, 200:-100])
# plt.savefig('/home/nmonnier/Papiers/JWST_MRS_Fusion/figures/mc_map.png', dpi=800)


