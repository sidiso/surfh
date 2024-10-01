import numpy as np

from skimage.measure import label
from surfh.ToolsDir import shepard_interpolation
from astropy.io import fits
import matplotlib.pyplot as plt

from rich import print
from rich.progress import track
from rich.console import Console

console = Console()

def get_fits_target_coordinates(filepath):
    """
    Extract target coordinates from FITS header.
    """
    with fits.open(filepath) as hdul:
        hdr = hdul[1].header
        targ_ra = hdr['RA_V1']
        targ_dec = hdr['DEC_V1']

    return targ_ra, targ_dec


def generate_label_image(binary_grid):
    """
    Generate and label connected components from the binary grid.
    """
    console.log("[cyan]Generating label image for connected components...[/cyan]")
    label_image = label(binary_grid)
    console.log(f"[green]Label image generated with {label_image.max()} components.[/green]")
    return label_image


def sort_labels_by_centroid(label_image):
    """
    Sort the labels by the x-coordinate of the centroids.
    """
    console.log("[cyan]Sorting labels by centroid coordinates...[/cyan]")
    from scipy.ndimage import center_of_mass
    num_labels = label_image.max()
    centroids = center_of_mass(label_image, label_image, range(1, num_labels + 1))
    sorted_labels = np.argsort([centroid[1] for centroid in centroids]) + 1

    # Create a new labeled image with sorted labels
    sorted_labeled_image = np.zeros_like(label_image)
    for new_label, old_label in enumerate(sorted_labels, start=1):
        sorted_labeled_image[label_image == old_label] = new_label

    console.log("[green]Labels sorted successfully.[/green]")
    return sorted_labeled_image


def perform_shepard_interpolation(alpha_valid, lambda_valid, 
                                  intensity_valid, 
                                  alpha_mesh, lambda_mesh, 
                                  p, alpha_exp, pixel_cutoff, 
                                  alpha_res, lambda_res):
    """
    Perform Shepard interpolation on the provided grid.

    Parameters
    ----------
    alpha_valid: 1D - array-like
      alpha coordinated of the data to be interpolated.
    lambda_valid: 1D - array-like
      lambda coordinated of the data to be interpolated.
    intensity_valid: 1D - array-like
      Intensity of the data to be interpolated.
    alpha_mesh: 2D - array-like
      alpha mesh coordinated of the output grid.
    lambda_mesh: 2D - array-like
      alpha mesh coordinated of the output grid.
    p : float
      The power for inverse distance weighting.
    alpha_exp : float
      The exponential decay factor.
    pixel_cutoff : float  
      The cutoff radius in pixels for influence. Points further than this cutoff will not be considered.
    alpha_res : float 
      The resolution in the alpha direction (e.g., pixel width).
    lambda_res : float
      The resolution in the lambda direction (e.g., pixel height).

    Returns
    -------
    interpolated_values: 2D - array-like
      Interpolated grid.

    """
    console.log("[cyan]Starting Shepard interpolation...[/cyan]")
    interpolated_values = shepard_interpolation.exponential_modified_shepard(
        alpha_valid.astype(np.float32), lambda_valid.astype(np.float32), intensity_valid.astype(np.float32),
        alpha_mesh.astype(np.float32), lambda_mesh.astype(np.float32), p=p, alpha=alpha_exp,
        pixel_cutoff=pixel_cutoff, alpha_res=alpha_res, lambda_res=lambda_res
    )
    console.log(f"[green]Interpolation completed...[/green]")
    
    return interpolated_values





def mrs_slices_distrorsion_correction(model_channel, sorted_labeled_image, detector2world, data, chan_wavelength, mode):
    """
    Visualize the interpolated image using labeled connected components.

    Parameters
    ----------
    model_channel : 1D - array-like
      alpha coordinated of the data to be interpolated.
    sorted_labeled_image: 1D - array-like
      lambda coordinated of the data to be interpolated.
    detector2world: 1D - array-like
      Intensity of the data to be interpolated.
    data: 2D - array-like
      alpha mesh coordinated of the output grid.
    chan_wavelength: 2D - array-like
      alpha mesh coordinated of the output grid.

    Returns
    -------
    interpolated_values: 2D - array-like
      Interpolated grid.


    """
    console.log("[cyan]Visualizing labeled image...[/cyan]")
    corrected_slices = np.zeros(model_channel.oshape[1:]) # First dim of model_channel is the number of obs

    i = 0
    for slit in range(len(np.unique(sorted_labeled_image))):
        
        if slit == 0:
            continue
        
        pixel_set = np.where(sorted_labeled_image == slit)
        alpha, beta, lam = detector2world(pixel_set[1], pixel_set[0])

        if mode == 0:
          if np.any(lam > np.max(chan_wavelength) +1):
              console.log(f"[yellow]Skipping slit due to wavelength limits.[/yellow]")
              continue
        
        if mode == 1:
          if np.any(lam < np.min(chan_wavelength) -1):
              console.log(f"[yellow]Skipping slit due to wavelength limits.[/yellow]")
              continue
        

        intensity = data[pixel_set]

        # Remove points where intensity is nan
        valid_mask = ~np.isnan(intensity)
        alpha_valid = alpha[valid_mask]
        lambda_valid = lam[valid_mask]
        intensity_valid = intensity[valid_mask]


        surfh_alpha_coordinates = np.linspace(np.min(alpha), np.max(alpha), model_channel.oshape[-1])
        alpha_mesh, lambda_mesh = np.meshgrid(surfh_alpha_coordinates, chan_wavelength)

        # Define pixel resolution in both directions (based on your grid structure)
        surfh_lambda_coordinates = chan_wavelength
        alpha_res = (np.max(surfh_alpha_coordinates) - np.min(surfh_alpha_coordinates)) / alpha_mesh.shape[1]
        lambda_res = (np.max(surfh_lambda_coordinates) - np.min(surfh_lambda_coordinates)) / lambda_mesh.shape[0]

        # Create meshgrid and perform interpolation
        console.log(f"[cyan]Performing interpolation for slit {i}...[/cyan]")
        corrected_slices[i] = perform_shepard_interpolation(alpha_valid, lambda_valid, intensity_valid, alpha_mesh, lambda_mesh, 2, 2.0, 2, alpha_res, lambda_res)
        i += 1

    console.log("[green]Visualization complete![/green]")

    return corrected_slices



