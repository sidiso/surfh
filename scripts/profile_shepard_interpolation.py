import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle

from surfh.Simulation import simulation_data

from surfh.Models import instru

from surfh.Models import wavelength_mrs, realmiri

from jwst import datamodels

from skimage.measure import label






@profile
def pixel_distance(p1, p2, alpha_res, lambda_res):
    """
    Compute the pixel distance between two points, scaled according to the resolution of alpha and lambda.
    
    Parameters:
    p1, p2: The two points between which the distance is computed (each is a tuple of (alpha, lambda)).
    alpha_res: The resolution (pixel size) in the alpha direction.
    lambda_res: The resolution (pixel size) in the lambda direction.
    
    Returns:
    The distance in pixel units.
    """
    dist1 = (p1[0] - p2[0]) / alpha_res
    dist2 = (p1[1] - p2[1]) / lambda_res
    dist1_square = dist1**2
    dist2_square = dist2**2
    dist = np.sqrt(dist1_square + dist2_square)
    return dist

@profile
def exponential_weight(dist, p=2, alpha=2.0):
    """
    Compute the exponential weight for a given distance.
    
    Parameters:
    dist: The pixel distance between points.
    p: The power to raise the distance to (typically 2 for inverse distance).
    alpha: The exponential factor that controls the steepness.
    
    Returns:
    The weight associated with the distance.
    """
    value = -alpha * dist**p
    exp = np.exp(value)
    return exp

@profile
def exponential_modified_shepard(points, values, query_points, p=2, alpha=2.0, pixel_cutoff=None, alpha_res=1.0, lambda_res=1.0, epsilon=1e-6):
    """
    Apply the Exponential Modified-Shepard interpolation method over a grid.
    
    Parameters:
    points: Tuple of (alpha, lambda) valid coordinates.
    values: Values at the known data points (intensity).
    query_points: Tuple of meshgrid (alpha_mesh, lambda_mesh) for interpolation.
    p: The power for inverse distance weighting (default is 2).
    alpha: The exponential decay factor (default is 2.0).
    pixel_cutoff: The cutoff radius in pixels for influence. Points further than this cutoff will not be considered.
    alpha_res: The resolution in the alpha direction (e.g., pixel width).
    lambda_res: The resolution in the lambda direction (e.g., pixel height).
    epsilon: A small value to avoid division by zero.
    
    Returns:
    Interpolated intensity values for the entire grid.
    """
    alpha_mesh, lambda_mesh = query_points
    interpolated_values = np.zeros_like(alpha_mesh)
    
    # Iterate over each point in the grid
    for i in range(alpha_mesh.shape[0]):
        for j in range(alpha_mesh.shape[1]):
            query_point = (alpha_mesh[i, j], lambda_mesh[i, j])
            
            # Initialize numerator and denominator for Shepard's method
            numerator = 0.0
            denominator = 0.0
            
            # Iterate over known points and compute weights
            for k in range(len(values)):
                pt = (points[0][k], points[1][k])
                dist = pixel_distance(pt, query_point, alpha_res, lambda_res) + epsilon  # Avoid division by zero
                
                # Apply pixel cutoff if specified
                if pixel_cutoff is None or dist <= pixel_cutoff:
                    w = exponential_weight(dist, p=p, alpha=alpha)
                    
                    numerator += w * values[k]
                    denominator += w
            
            # Store interpolated value at this grid point
            interpolated_values[i, j] = numerator / denominator if denominator != 0 else 0.0
    
    return interpolated_values







def main():
    """
    Create Model and simulation
    """
    origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(4, 0, '~/Projects/JWST/MRS/surfh/cube_orion/') # subsampling to reduce dim of maps

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
    test = detector2world(120,120)
    binary_grid = np.zeros_like(data)


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

    # Perform interpolation on the grid
    interpolated_values = exponential_modified_shepard(
        points, values, query_points, p=p, alpha=alpha_exp, pixel_cutoff=pixel_cutoff, alpha_res=alpha_res, lambda_res=lambda_res
    )

    print("Interpolated intensity values:\n", interpolated_values)

if __name__ == '__main__':
    main()