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
from surfh.DottestModels import MCMO_SigRLSCT_Channel_Model
import time
from rich import print
from rich.progress import track
from rich.console import Console

console = Console()

def load_simulation_data():
    """
    Load simulation data and the wavelength information.
    """
    console.log("[bold cyan]Loading simulation data...[/bold cyan]")
    origin_alpha_axis, origin_beta_axis, wavel_axis, spsf, maps, templates = simulation_data.get_simulation_data(4, 0, '/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/')
    console.log("[bold cyan]Loading wavelength data...[/bold cyan]")
    wavelength_1c = wavelength_mrs.get_mrs_wavelength('1c')
    
    console.log("[green]Simulation data and wavelength loaded successfully![/green]")
    return origin_alpha_axis, origin_beta_axis, wavelength_1c


def get_fits_data(filepath):
    """
    Extract target coordinates from FITS header.
    """
    console.log(f"[bold cyan]Opening FITS file: {filepath}[/bold cyan]")
    with fits.open(filepath) as hdul:
        hdr = hdul[1].header
        targ_ra = hdr['RA_V1']
        targ_dec = hdr['DEC_V1']
    console.log(f"[green]Extracted RA: {targ_ra}, DEC: {targ_dec} from FITS file.[/green]")

    return targ_ra, targ_dec


def setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ch1c, wavelength_1c):
    """
    Set up channel model with super resolution and global wavelength.
    """
    console.log("[bold cyan]Setting up channel model...[/bold cyan]")
    step = 0.025
    step_angle = Angle(step, u.arcsec).degree
    ch1c_pix = ch1c.pix(step_angle)

    console.log("[cyan]Calculating super resolution factor...[/cyan]")
    super_resolution_factor = instru.get_srf(
        [ch1c.det_pix_size],
        step_angle * 3600,  # Conversion in arcsec
    )

    console.log("[cyan]Adjusting alpha and beta axes based on target RA and DEC...[/cyan]")
    alpha_axis = origin_alpha_axis - np.mean(origin_alpha_axis) + targ_ra
    beta_axis = origin_beta_axis - np.mean(origin_beta_axis) + targ_dec
    pointings = instru.CoordList([instru.Coord(targ_ra, targ_dec)]).pix(step_angle)

    console.log("[cyan]Extending the wavelength axis...[/cyan]")
    a = wavelength_1c[0] - np.flip((np.arange(10) + 1)) * (wavelength_1c[1] - wavelength_1c[0]) * 10
    b = wavelength_1c[-1] + (np.arange(10) + 1) * (wavelength_1c[-1] - wavelength_1c[-2]) * 10
    global_wavelength = np.concatenate([a, wavelength_1c, b])

    console.log("[cyan]Creating channel model...[/cyan]")
    channel = MCMO_SigRLSCT_Channel_Model.Channel(
        ch1c,
        alpha_axis,
        beta_axis,
        global_wavelength,
        super_resolution_factor[0],
        pointings,
        step_angle
    )
    console.log("[green]Channel model setup completed![/green]")
    return channel


def perform_interpolation(alpha_valid, lambda_valid, intensity_valid, alpha_mesh, lambda_mesh, p, alpha_exp, pixel_cutoff, alpha_res, lambda_res):
    """
    Perform Shepard interpolation on the provided grid.
    """
    console.log("[cyan]Starting Shepard interpolation...[/cyan]")
    start = time.time()
    interpolated_values = shepard_interpolation.exponential_modified_shepard(
        alpha_valid.astype(np.float32), lambda_valid.astype(np.float32), intensity_valid.astype(np.float32),
        alpha_mesh.astype(np.float32), lambda_mesh.astype(np.float32), p=p, alpha=alpha_exp,
        pixel_cutoff=pixel_cutoff, alpha_res=alpha_res, lambda_res=lambda_res
    )
    end = time.time()
    console.log(f"[green]Interpolation completed in {end - start:.2f} seconds.[/green]")
    
    return interpolated_values


def process_fits_data(model, pixel_set):
    """
    Extract coordinates and intensity from the FITS file for a given pixel set.
    """
    console.log(f"[cyan]Processing FITS data for pixel set with {len(pixel_set[0])} points...[/cyan]")
    detector2world = model.meta.wcs.get_transform('detector', 'world')
    alpha, beta, lam = detector2world(pixel_set[1], pixel_set[0])
    intensity = model.data[pixel_set]
    console.log("[green]FITS data processed successfully.[/green]")

    return alpha, lam, intensity


def plot_interpolated_data(alpha, lam, intensity):
    """
    Create a scatter plot for the interpolated data.
    """
    console.log("[cyan]Plotting interpolated data...[/cyan]")
    sc = plt.scatter(alpha, lam, c=intensity)
    plt.colorbar(sc)
    plt.title(f"Real data 2D Scatter of slice n°{0}")
    plt.show()


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


def visualize_labeled_image(channels, sorted_labeled_image, detector2world, data, wavelength_1c):
    """
    Visualize the interpolated image using labeled connected components.
    """
    console.log("[cyan]Visualizing labeled image...[/cyan]")
    n = channels[0].oshape[1]
    space = 0.1
    fig, ax = plt.subplots(figsize=(n * 1, 32))
    ax.axis('off')
    vmin, vmax = 0, 25000

    for slit in track(range(len(np.unique(sorted_labeled_image))), description="Processing slits"):
        if slit == 0:
            continue
        pixel_set = np.where(sorted_labeled_image == slit)
        alpha, beta, lam = detector2world(pixel_set[1], pixel_set[0])
        if np.any(lam > 9):
            console.log(f"[yellow]Skipping slit {slit} due to wavelength limits.[/yellow]")
            continue

        intensity = data[pixel_set]

        # Remove points where intensity is nan
        valid_mask = ~np.isnan(intensity)
        alpha_valid = alpha[valid_mask]
        lambda_valid = lam[valid_mask]
        intensity_valid = intensity[valid_mask]


        surfh_alpha_coordinates = np.linspace(np.min(alpha), np.max(alpha), channels[0].oshape[-1])
        alpha_mesh, lambda_mesh = np.meshgrid(surfh_alpha_coordinates, wavelength_mrs.get_mrs_wavelength('1c'))

        # Define pixel resolution in both directions (based on your grid structure)
        surfh_lambda_coordinates = wavelength_1c
        alpha_res = (np.max(surfh_alpha_coordinates) - np.min(surfh_alpha_coordinates)) / alpha_mesh.shape[1]
        lambda_res = (np.max(surfh_lambda_coordinates) - np.min(surfh_lambda_coordinates)) / lambda_mesh.shape[0]

        # Create meshgrid and perform interpolation
        console.log(f"[cyan]Performing interpolation for slit {slit}...[/cyan]")
        intensity_grid = perform_interpolation(alpha_valid, lambda_valid, intensity_valid, alpha_mesh, lambda_mesh, 2, 2.0, 2, alpha_res, lambda_res)

        left = slit - 1 + (slit - 1) * space
        right = left + 1
        height = len(wavelength_mrs.get_mrs_wavelength('1c'))
        im = ax.imshow(np.flipud(intensity_grid), extent=[left, right, 0, height], aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    ax.set_xlim(0, n + (n - 1) * space)
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Intensity')

    console.log("[green]Visualization complete![/green]")
    plt.show()


def main():
    console.log("[bold blue]--- Starting process ---[/bold blue]")
    
    # Load simulation data
    console.log("[bold cyan]Loading simulation data...[/bold cyan]")
    origin_alpha_axis, origin_beta_axis, wavelength_1c = load_simulation_data()

    # Extract target coordinates from FITS file
    console.log("[bold cyan]Extracting target coordinates from FITS file...[/bold cyan]")
    targ_ra, targ_dec = get_fits_data('/home/nmonnier/Data/JWST/Orion_bar/Stage_2/jw01288002001_0211f_00001_mirifushort_cal.fits')
    
    # Setup channel model
    console.log("[bold cyan]Setting up channel model...[/bold cyan]")
    ch1c, targ_ra, targ_dec = realmiri.get_IFU('/home/nmonnier/Data/JWST/Orion_bar/Stage_3/ChannelCube_ch1-long_s3d.fits')
    channel = setup_channel_model(origin_alpha_axis, origin_beta_axis, targ_ra, targ_dec, ch1c, wavelength_1c)

    # Process FITS data and generate labeled image
    console.log("[bold cyan]Processing FITS data and generating labeled image...[/bold cyan]")
    model = datamodels.open('/home/nmonnier/Data/JWST/Orion_bar/Stage_2/jw01288002001_0211f_00001_mirifushort_cal.fits')
    data = model.data

    binary_grid = np.zeros_like(data)
    x_pixel_idx = np.arange(data.shape[0])
    y_pixel_idx = np.arange(data.shape[1])
    yy, xx = np.meshgrid(x_pixel_idx, y_pixel_idx)

    detector2world = model.meta.wcs.get_transform('detector', 'world')
    coordinates = detector2world(xx, yy)
    binary_grid[~np.isnan(coordinates[0].T)] = 1

    console.log("[bold cyan]Generating label image...[/bold cyan]")
    label_image = generate_label_image(binary_grid)

    # Sort labels and visualize
    console.log("[bold cyan]Sorting labels by centroid...[/bold cyan]")
    sorted_labeled_image = sort_labels_by_centroid(label_image)

    console.log("[bold cyan]Visualizing labeled image...[/bold cyan]")
    visualize_labeled_image([channel], sorted_labeled_image, detector2world, data, wavelength_1c)

    console.log("[bold blue]--- Process completed successfully! ---[/bold blue]")


if __name__ == "__main__":
    main()





