import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

cube_mrs_path = '/home/nmonnier/Data/JWST/NGC_7023/Scan/NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits'
filter_miri_path = '/home/nmonnier/Data/JWST/NGC_7023/Scan/Level3_F1800W_i2d.fits'

# ----- Cube MRS -----
with fits.open(cube_mrs_path) as hdul:
    sci_hdu = hdul[hdul.index_of("SCI")]
    mrs_data = sci_hdu.data.copy()
    hdr = sci_hdu.header

ny, nx = mrs_data.shape[1], mrs_data.shape[2]

# ----- WCS 2D spatial propre -----
mrs_wcs = WCS(naxis=2)
mrs_wcs.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
mrs_wcs.wcs.cdelt = [hdr['CDELT1'], hdr['CDELT2']]
mrs_wcs.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
mrs_wcs.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]

# ----- Image MIRI -----
with fits.open(filter_miri_path) as hdul:
    miri_hdu = hdul[hdul.index_of("SCI")]
    miri_data = miri_hdu.data.copy()
    miri_wcs = WCS(miri_hdu.header).celestial

# ----- Slice du cube -----
slice_index = 2000
mrs_slice = mrs_data[slice_index, :, :]

# ----- Reprojection -----
mrs_reproj, footprint = reproject_interp(
    (mrs_slice, mrs_wcs),
    miri_wcs,
    shape_out=miri_data.shape
)

mrs_norm = mrs_reproj / np.nanmax(mrs_reproj)

# ----- Plot -----
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(projection=miri_wcs)
ax.imshow(miri_data, origin='lower', cmap='gray')
ax.imshow(mrs_norm, origin='lower', cmap='inferno', alpha=0.5)
ax.set_xlabel("RA")
ax.set_ylabel("Dec")
plt.title(f"MIRI + MRS slice {slice_index}")
plt.show()
