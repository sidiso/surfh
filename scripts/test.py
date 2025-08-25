import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from matplotlib.patches import Polygon



# Get all FITS files

dir = "/home/nmonnier/Data/JWST/PIPELINE/small_NGC/stage2/"
fits_files = [
    dir + "jw01192001001_0310v_00001_mirifushort_s3d.fits",
    dir + "jw01192001001_0310v_00002_mirifushort_s3d.fits",
    dir + "jw01192001001_0310v_00003_mirifushort_s3d.fits",
    dir + "jw01192001001_0310v_00004_mirifushort_s3d.fits",
]
# dir = "/home/nmonnier/Data/JWST/PIPELINE/Point_source/stage2/"
# fits_files = [
#     dir + "jw06659001001_05101_00001_mirifushort_s3d.fits",
#     dir + "jw06659001001_05101_00002_mirifushort_s3d.fits",
#     dir + "jw06659001001_05101_00003_mirifushort_s3d.fits",
#     dir + "jw06659001001_05101_00004_mirifushort_s3d.fits",
# ]
# Select a reference file (first FITS file)
ref_file = fits_files[0]
with fits.open(ref_file) as ref_hdu:
    ref_wcs = WCS(ref_hdu[1].header, ref_hdu)  # Pass HDUList to handle -TAB coordinates
    ref_shape = (ref_hdu[1].header["NAXIS2"], ref_hdu[1].header["NAXIS1"])  # 2D shape

# Initialize a figure with WCS projection (fix: specify slices for 2D projection)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ref_wcs, "slices": ("x", "y", 0)})

i=0
# Loop over FITS files
for file in fits_files:
    with fits.open(file) as hdul:
        wcs_in = WCS(hdul[1].header, hdul)  # Pass HDUList
        data_in = hdul[1].data  # 3D data cube
        data_in[np.isnan(data_in)] = 0  # Replace NaNs with zeros

        # Extract a single slice (e.g., middle slice along NAXIS3)
        slice_idx = -1 #data_in.shape[0] // 2  # Middle wavelength slice
        slice_2d = data_in[slice_idx, :, :]  # Shape: (NAXIS2, NAXIS1)

        print("Min:", np.nanmin(slice_2d), "Max:", np.nanmax(slice_2d))
        print("Mean:", np.nanmean(slice_2d))
        print("Shape:", slice_2d.shape)

        # Reproject slice onto the reference WCS
        shape_out_3d = (1,) + ref_shape  # Ensure it's 3D
        data_out, _ = reproject_interp((slice_2d[np.newaxis, :, :], wcs_in), ref_wcs, shape_out=shape_out_3d)
        data_out = data_out[0]  # Convert back to 2D for plotting

        data_out = np.nan_to_num(data_out, nan=0)

        print("Min:", np.nanmin(data_out), "Max:", np.nanmax(data_out))
        print("Mean:", np.nanmean(data_out))
        print("Shape:", data_out.shape)

        # Plot the reprojected slice
        ax.imshow(data_out, origin="lower", cmap="plasma", alpha=0.5)  # Adjust transparency

        # Get CRPIX values from the header
        crpix1, crpix2, crpix3 = hdul[1].header["CRPIX1"], hdul[1].header["CRPIX2"], hdul[1].header["CRPIX3"]

        # Convert (CRPIX1, CRPIX2, CRPIX3) to world coordinates (RA, Dec, Wavelength)
        ra_center, dec_center, wave_center = wcs_in.all_pix2world(crpix1, crpix2, crpix3, 0)
        print("RA:", ra_center, "Dec:", dec_center)

        # Convert (RA, Dec, Wavelength) to pixel coordinates in reference WCS
        x_center, y_center, _ = ref_wcs.all_world2pix(ra_center, dec_center, wave_center, 0)
        print("X:", x_center, "Y:", y_center)
        # Plot a cross at the center
        ax.plot(x_center, y_center, marker="+", markersize=10, markeredgewidth=2, color="white", label="Projection Center")

        # Define the four corners of the image in pixel coordinates
        image_shape = data_in.shape  # Shape of the FITS image
        height, width = image_shape[1], image_shape[2]  # NAXIS2, NAXIS1
        print(f'height: {height}, width: {width}')

        corners_pix = np.array([
            [0, 0],        # Bottom-left
            [0, width-1],  # Bottom-right
            [height-1, width-1],  # Top-right
            [height-1, 0]   # Top-left
        ])

        # Convert pixel coordinates to world coordinates (RA, Dec)
        corners_world = wcs_in.all_pix2world(corners_pix[:, 0], corners_pix[:, 1], np.zeros(4), 0)
        print("Corners World Coordinates:", corners_world)

        # Convert world coordinates to pixel coordinates in the reference WCS
        corners_ref_pix = np.array(ref_wcs.all_world2pix(corners_world[0], corners_world[1], np.zeros(4), 0))

        # Only take the first two columns (X, Y)
        corners_ref_pix = corners_ref_pix[:2].T  # Transpose to get (M,2) shape
        print("Corners Reference Pixel Coordinates:", corners_ref_pix)

        # Plot the polygon outline
        polygon = Polygon(corners_ref_pix, edgecolor="cyan", facecolor="none", linewidth=2)
        ax.add_patch(polygon)
        i = i+1
    

# x_center, y_center, _ = ref_wcs.all_world2pix(315.38216470139986, 68.1737618072801, wave_center, 0)
# ax.plot(x_center, y_center, marker="+", markersize=10, markeredgewidth=2, color="red", label="Projection Center")
# x_center, y_center, _ = ref_wcs.all_world2pix(315.3818968109811, 68.17402533111624, wave_center, 0)
# ax.plot(x_center, y_center, marker="+", markersize=10, markeredgewidth=2, color="red", label="Projection Center")
# x_center, y_center, _ = ref_wcs.all_world2pix(315.38233680667923, 68.17380958724748, wave_center, 0)
# ax.plot(x_center, y_center, marker="+", markersize=10, markeredgewidth=2, color="red", label="Projection Center")
# x_center, y_center, _ = ref_wcs.all_world2pix(315.38172580550366, 68.17397786701078, wave_center, 0)
# ax.plot(x_center, y_center, marker="+", markersize=10, markeredgewidth=2, color="red", label="Projection Center")

# Labels and grid
ax.set_xlabel("Right Ascension")
ax.set_ylabel("Declination")
ax.grid(color="white", ls="dotted")

# Show the final overlaid image
plt.show()
