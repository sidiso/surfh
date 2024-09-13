import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle

from surfh.Simulation import simulation_data
from surfh.Models import instru

from surfh.Models import wavelength_mrs, realmiri
from surfh.ToolsDir import shepard_interpolation

from jwst import datamodels

from skimage.measure import label
import time

def main():
    """
    Create Model and simulation
    """
    origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(4, 0, '/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/') # subsampling to reduce dim of maps

    wavelength_1c = wavelength_mrs.get_mrs_wavelength('1c')
    # n = len(wavelength_1c)
    # x_original = np.arange(n)
    # x_oversampled = np.linspace(0, n - 1, 2 * n - 1)
    # oversampled_vector = np.interp(x_oversampled, x_original, original_vector)


    with fits.open('/home/nmonnier/Data/JWST/Orion_bar/Stage_2/jw01288002001_0211f_00001_mirifushort_cal.fits') as hdul:
        hdr = hdul[1].header
        targ_ra  = hdr['RA_V1']
        targ_dec = hdr['DEC_V1']


    orgin_target = instru.Coord(targ_ra, targ_dec)
    ch1c, targ_ra, targ_dec = realmiri.get_IFU('/home/nmonnier/Data/JWST/Orion_bar/Stage_3/ChannelCube_ch1-long_s3d.fits')
    step = 0.025
    step_angle = Angle(step, u.arcsec).degree
    ch1c_pix = ch1c.pix(step_angle)

    # Super resolution factor (in alpha dim) 
    super_resolution_factor = instru.get_srf(
        [ch1c.det_pix_size],
        step_angle*3600, # Conversion in arcsec
    )

    alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + targ_ra
    beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + targ_dec
    pointings = instru.CoordList([orgin_target]).pix(step_angle)
    from surfh.DottestModels import MCMO_SigRLSCT_Channel_Model
    # Channel 
    a = wavelength_1c[0] - np.flip((np.arange(10)+1))*(wavelength_1c[1]-wavelength_1c[0])*10
    b = wavelength_1c[-1] + (np.arange(10)+1)*(wavelength_1c[-1]-wavelength_1c[-2])*10

    global_wavelength = np.copy(wavelength_1c)
    global_wavelength = np.append(global_wavelength, b)
    global_wavelength = np.insert(global_wavelength, 0, a)

    print(f'Wavelength resolution is {wavelength_1c[1]-wavelength_1c[0]}')
    print("wavelength_1c size is", wavelength_1c.shape)
    print("Global wavelenfth size is", global_wavelength.shape)
    print(wavelength_1c)
    print(global_wavelength)
    channels = [
                MCMO_SigRLSCT_Channel_Model.Channel(
                    ch1c,
                    alpha_axis,
                    beta_axis,
                    global_wavelength,
                    super_resolution_factor[0],
                    pointings,
                    step_angle
                )
            ]

    print(f'Slit O-Shape channel 1C : {channels[0].oshape}')
    print(f'Slit I-Shape channel 1C : {channels[0].ishape}')
    print(ch1c.slit_fov[1].local.alpha_width)
    print(f'Shape of Channel 1C output is ')



    model=datamodels.open('/home/nmonnier/Data/JWST/Orion_bar/Stage_2/jw01288002001_0211f_00001_mirifushort_cal.fits')
    data = model.data

    x_shape = data.shape[0]
    y_shape = data.shape[1]

    x_pixel_idx = np.arange(x_shape)
    y_pixel_idx = np.arange(y_shape)

    yy, xx = np.meshgrid(x_pixel_idx, y_pixel_idx)

    detector2world = model.meta.wcs.get_transform('detector', 'world')

    coordinates = detector2world(xx, yy)
    binary_grid = np.zeros_like(data)

    from skimage.measure import label
    binary_grid[~np.isnan(coordinates[0].T)] = 1
    # Label connected components: Identify and label the white regions
    label_image = label(binary_grid)

    pixel_set = np.where(label_image==10)

    alpha, beta, lam = detector2world(pixel_set[1],pixel_set[0])

    intensity = data[pixel_set]
    #plt.figure(figsize=(2, 8))
    sc = plt.scatter(alpha, lam, c=intensity)
    plt.colorbar(sc)
    plt.title(f"Real data 2D Scatter of slice n°{0}")

    surfh_alpha_coordinates = np.linspace(np.min(alpha), np.max(alpha), channels[0].oshape[-1] + 1)


    world2detector = model.meta.wcs.get_transform('world', 'detector')
    world2detector(alpha[0], beta[0], lam[0])
    print(f'Min alpha is {np.min(alpha)}, Max alpha is {np.max(alpha)}')
    print(f'Range alpha is {np.max(alpha) - np.min(alpha)}')


    # Remove points where intensity is nan
    valid_mask = ~np.isnan(intensity)
    alpha_valid = alpha[valid_mask]
    lambda_valid = lam[valid_mask]
    intensity_valid = intensity[valid_mask]


    intensity = data[pixel_set]
    alpha_valid = alpha[valid_mask]
    lambda_valid = lam[valid_mask]
    intensity_valid = intensity[valid_mask]

    points = (alpha_valid, lambda_valid)
    values = intensity_valid

    surfh_alpha_coordinates = np.linspace(np.min(alpha), np.max(alpha), channels[0].oshape[-1])
    surfh_lambda_coordinates = wavelength_1c

    alpha_mesh, lambda_mesh = np.meshgrid(surfh_alpha_coordinates, surfh_lambda_coordinates)
    query_points = (alpha_mesh, lambda_mesh)

    # Set parameters for interpolation
    p = 2            # Power for inverse distance weighting
    alpha_exp = 2.0  # Exponential decay factor
    pixel_cutoff = 2 # Radius cutoff in pixels

    # Define pixel resolution in both directions (based on your grid structure)
    alpha_res = (np.max(surfh_alpha_coordinates) - np.min(surfh_alpha_coordinates)) / alpha_mesh.shape[1]
    lambda_res = (np.max(surfh_lambda_coordinates) - np.min(surfh_lambda_coordinates)) / lambda_mesh.shape[0]

    start = time.time()
    # Perform interpolation on the grid
    interpolated_values = shepard_interpolation.exponential_modified_shepard(
        alpha_valid.astype(np.float32), lambda_valid.astype(np.float32), values.astype(np.float32), alpha_mesh.astype(np.float32), lambda_mesh.astype(np.float32), p=p, alpha=alpha_exp, pixel_cutoff=pixel_cutoff, alpha_res=alpha_res, lambda_res=lambda_res
    )
    end = time.time()

    print("Computation time is ", end-start)
    print("Interpolated intensity values:\n", interpolated_values)

    plt.figure(figsize=(8,8))

    plt.imshow(np.flipud(interpolated_values), aspect = 0.2)
    plt.colorbar()
    plt.show()






    from scipy.ndimage import label, find_objects, center_of_mass

    # Get the number of labels and label slices
    num_labels = label_image.max()

    # Compute the center of mass for each labeled region
    centroids = center_of_mass(label_image, label_image, range(1, num_labels + 1))

    # Sort labels by the x-coordinate (column index) of the centroids
    sorted_labels = np.argsort([centroid[1] for centroid in centroids]) + 1

    # Create a new labeled image with the labels sorted from left to right
    sorted_labeled_image = np.zeros_like(label_image)
    for new_label, old_label in enumerate(sorted_labels, start=1):
        sorted_labeled_image[label_image == old_label] = new_label


    # Number of images (bars)
    n = channels[0].oshape[1]

    # Space between images (as a fraction of image width)
    space = 0.1

    # Create a figure with a specified width and height
    fig, ax = plt.subplots(figsize=(n * 1, 32))

    # Set the axis off
    ax.axis('off')
    vmin = 0
    vmax = 25000

    list_intensity_grid = []
    for slit in range(len(np.unique(sorted_labeled_image))):
        print("-----------------------------------")
        if slit == 0:
            continue
        pixel_set = np.where(sorted_labeled_image==slit)
        alpha, beta, lam = detector2world(pixel_set[1],pixel_set[0])
        if np.any(lam > 9):
            continue
        
        print(f"Idx = {slit}")
        list_alpha = []
        list_beta = []
        
        
        surfh_alpha_coordinates = np.linspace(np.min(alpha), np.max(alpha), channels[0].oshape[-1])
        step_size = surfh_alpha_coordinates[1] - surfh_alpha_coordinates[0]

        # Extend the vector
        extended_surfh_alpha_coordinates = np.concatenate((
            [surfh_alpha_coordinates[0] - step_size],  # Add one element before
            surfh_alpha_coordinates,                   # Original coordinates
            [surfh_alpha_coordinates[-1] + step_size]  # Add one element after
        ))
        print(f'Min alpha = {np.min(alpha)}, max alpha =  {np.max(alpha)}')
        print(f'Alpha = {alpha}')
        surfh_lambda_coordinates = wavelength_1c

        alpha, beta, lam = detector2world(pixel_set[1],pixel_set[0])
        intensity = data[pixel_set]


        # Create a meshgrid from the output grid
        alpha_mesh, lambda_mesh = np.meshgrid(extended_surfh_alpha_coordinates, surfh_lambda_coordinates)

        # Remove points where intensity is nan
        valid_mask = ~np.isnan(intensity)
        alpha_valid = alpha[valid_mask]
        lambda_valid = lam[valid_mask]
        intensity_valid = intensity[valid_mask]

        # Interpolate the data onto the grid
        points = (alpha_valid, lambda_valid)
        values = intensity_valid

        surfh_alpha_coordinates = np.linspace(np.min(alpha), np.max(alpha), channels[0].oshape[-1])
        surfh_lambda_coordinates = wavelength_1c

        alpha_mesh, lambda_mesh = np.meshgrid(surfh_alpha_coordinates, surfh_lambda_coordinates)

        # Set parameters for interpolation
        p = 2            # Power for inverse distance weighting
        alpha_exp = 2.0  # Exponential decay factor
        pixel_cutoff = 2 # Radius cutoff in pixels

        # Define pixel resolution in both directions (based on your grid structure)
        alpha_res = (np.max(surfh_alpha_coordinates) - np.min(surfh_alpha_coordinates)) / alpha_mesh.shape[1]
        lambda_res = (np.max(surfh_lambda_coordinates) - np.min(surfh_lambda_coordinates)) / lambda_mesh.shape[0]


        intensity_grid = shepard_interpolation.exponential_modified_shepard(
                                            alpha_valid.astype(np.float32), lambda_valid.astype(np.float32), 
                                            values.astype(np.float32), alpha_mesh.astype(np.float32), 
                                            lambda_mesh.astype(np.float32), p=p, 
                                            alpha=alpha_exp, pixel_cutoff=pixel_cutoff, 
                                            alpha_res=alpha_res, lambda_res=lambda_res
    )
        intensity_grid = intensity_grid[:,1:-1]
        list_intensity_grid.append(intensity_grid)



        # Calculate the distance between each point in alpha_valid and each point in surfh_alpha_coordinates
        distances = np.abs(alpha_valid[:, np.newaxis] - extended_surfh_alpha_coordinates)

        # Find the index of the nearest neighbor in surfh_alpha_coordinates for each point in alpha_valid
        nearest_neighbors = np.argmin(distances, axis=1)

        # Count how many times each surfh_alpha_coordinate is the nearest neighbor
        counts = np.bincount(nearest_neighbors, minlength=len(extended_surfh_alpha_coordinates))

        # Display the result
        for i, coord in enumerate(extended_surfh_alpha_coordinates):
            print(f'Coordinate {coord} has {counts[i]} nearest neighbors in alpha_valid.')




        # Calculate the left and right positions with space in between
        left = slit-1 + (slit-1) * space
        right = left + 1
        height = len(surfh_lambda_coordinates)  # height of the image
        im = ax.imshow(np.flipud(intensity_grid), extent=[left, right, 0, height], aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    ax.set_xlim(0, n + (n - 1) * space)
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Intensity')


    plt.show()

if __name__ == '__main__':
    main()