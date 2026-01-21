import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.ndimage import shift
from skimage.transform import rotate as ski_rotate
import scipy

from surfh.ToolsDir.alignment import interactive_align

def chi2(fusion_img, mrs_img, angle, y_shift, x_shift):
    fusion_img_rot = ski_rotate(fusion_img, angle=angle, resize=False, order=3, preserve_range=True)  
    fusion_img_shift = shift(fusion_img_rot, shift=(y_shift, x_shift), order=3)  
    fusion_img_masked = fusion_img_shift
    fusion_img_masked = fusion_img_masked
    return np.sum((fusion_img_masked - mrs_img)**2)

def get_result_image(fusion_img, angle, y_shift, x_shift):
    fusion_img_rot = ski_rotate(fusion_img, angle=angle, resize=False, order=3, preserve_range=True)
    fusion_img_shift = shift(fusion_img_rot, shift=(y_shift, x_shift), order=3)
    masked_fusion_img = fusion_img_shift
    return masked_fusion_img


ref_path = '/home/nmonnier/Data/JWST/NGC_7023/Scan/'
# fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_204_mu_5.00e+06_SD_True/'
fusion_path = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resultslcg_MC_11_MO_4_Temp_16_nit_1000_mu_5.00e+06_SD_True/'

# open reference FITS file
ref_file = ref_path + 'NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits'
with fits.open(ref_file) as ref_hdu:
    ref_data = ref_hdu[1].data
    ref_header = ref_hdu[1].header
ref_data[np.isnan(ref_data)] = 0

# open fusion result FITS file
fusion_file = fusion_path + 'res_cube.fits'
with fits.open(fusion_file) as fusion_hdu:
    fusion_data = fusion_hdu[0].data
    fusion_header = fusion_hdu[0].header

    wavelength = fusion_hdu['WCS-TABLE'].data['wavelength'][0]
    wavelength = wavelength.squeeze() # Because shape can be (N,1) or (1,N)

    PA = fusion_header['PA_V3']

masks = np.load(fusion_path + 'masks.npy')
ch_limit = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 24.19]
print("Masks shape:", masks.shape)                                                                                                                                                                                                                                                                                                       

# Pad spatial dimension of ref data to be the same shape as spacial dimension of fusion data
if ref_data.shape[1] < fusion_data.shape[1] or ref_data.shape[2] < fusion_data.shape[2]:
    pad_y = (fusion_data.shape[1] - ref_data.shape[1]) // 2
    pad_x = (fusion_data.shape[2] - ref_data.shape[2]) // 2
    ref_data = np.pad(ref_data, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    print(f"Padded ref data to shape: {ref_data.shape}")


sum_fusion_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
sum_ref_cube = np.zeros_like(masks, dtype=fusion_data.dtype)
print(f"SHape ref data: {sum_fusion_cube.shape}, fusion data: {fusion_data.shape}")
i = 0
idx_prev = 0 
for wave in ch_limit:
    idx_wavel = np.argmin(np.abs(wavelength - wave))
    print(f"Band {i}: Wavelength {wave} um, Index range: {idx_prev} to {idx_wavel}")
    
    fusion_data[idx_prev:idx_wavel] = fusion_data[idx_prev:idx_wavel,:,:]*masks[i]
    sum_fusion_cube[i] = np.nansum(fusion_data[idx_prev:idx_wavel,:,:], axis=0)


    sum_ref_cube[i] = np.nansum(ref_data[idx_prev:idx_wavel], axis=0)
    
    print(sum_fusion_cube.shape,fusion_data.shape,idx_prev, idx_wavel, i)
    idx_prev = idx_wavel
    i +=1



# Plot Both Data Cubes at a Specific Wavelength Index
band_idx = 6
print(PA)
fusion_slice = rotate(sum_fusion_cube[band_idx], angle=PA-180, reshape=False)
ref_slice = sum_ref_cube[band_idx]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f'Reference Data - Wavelength Index {band_idx}')
plt.imshow(ref_slice, origin='lower', cmap='inferno')
plt.colorbar(label='Intensity')
plt.subplot(1, 2, 2)
plt.title(f'Fusion Result Data - Wavelength Index {band_idx}')
plt.imshow(fusion_slice, origin='lower', cmap='inferno')
plt.colorbar(label='Intensity')
plt.tight_layout()


norm_fusion_data = fusion_slice/np.max(fusion_slice)
norm_ref_data = ref_slice/np.max(ref_slice)


fct_chi2 = lambda x : chi2(norm_fusion_data, norm_ref_data, angle=x[0], y_shift=x[1], x_shift=x[2])
result = scipy.optimize.minimize(
                        fct_chi2, 
                        (0, 0, 0), 
                        method='Nelder-Mead', 
                        options={'disp': True, 'maxiter': 2000, 'maxfev': 5000, 'xatol':1e-3, 'fatol':1e-3},
                            )

corrected_fusion_data = get_result_image(fusion_slice, *result.x)
print("Optimization result:", result)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title(f'Reference Data - Wavelength Index {band_idx}')
# plt.imshow(ref_slice, origin='lower', cmap='inferno')
# plt.colorbar(label='Intensity')
# plt.subplot(1, 2, 2)
# plt.title(f'Corrected Fusion Result Data - Wavelength Index {band_idx}')
# plt.imshow(corrected_fusion_data, origin='lower', cmap='inferno')
# plt.colorbar(label='Intensity')
# plt.tight_layout()

def ensure_native_endian(arr):
    if arr.dtype.byteorder not in ('=', '<'):
        arr = arr.byteswap().newbyteorder()
    return arr

print(f"Shape of images : ref {ref_slice.shape}, fusion {corrected_fusion_data.shape}")
ref_slice = ensure_native_endian(ref_slice)

corrected_fusion_data = ensure_native_endian(corrected_fusion_data)

interactive_align(ref_slice, corrected_fusion_data, np.ones_like(corrected_fusion_data))

plt.show()