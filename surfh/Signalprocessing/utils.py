import numpy as np
from scipy.ndimage import rotate, gaussian_filter, generic_filter

from surfh.Signalprocessing.baseline import iterative_baseline_removal
from surfh.Signalprocessing import fitting

def interpolate_negatives(cube):
    """
    Interpolate negative values using local positive neighborhood.
    
    Args:
        cube (np.ndarray): Input 3D array (wl, H, W).
    Returns:
        np.ndarray: Cube with negative values interpolated.
    """
    cube = cube.copy()

    def _interp(values):
        center = values[len(values) // 2]
        if center >= 0:
            return center
        positives = values[values >= 0]
        return np.mean(positives) if positives.size > 0 else 0.0

    return generic_filter(cube, _interp, size=3, mode="mirror")


def process_pixel_chunk(chunk):
    """
    Fit spectral lines pixel-by-pixel (parallelized).
    
    Args:
        chunk (list): List of tuples (i, j, cube, peak_indices, mean_sigma, std_sigma).
    Returns:
        list: List of results with fitted continuum and spectral lines for each pixel.
    """
    results = []

    for i, j, cube, peak_indices, mean_sigma, std_sigma in chunk:
        spectrum = cube[:, i, j]

        baseline = iterative_baseline_removal(
            spectrum, lam=1e3, ncycles=5, sigma=1.0
        )
        baseline_sub = spectrum - baseline

        fitted, continuum = fitting.fit_peaks_only(
            baseline_sub, spectrum,
            peak_indices, mean_sigma, std_sigma
        )

        x = np.arange(len(spectrum))
        spectral_line = np.zeros_like(spectrum)

        for peak in fitted:
            A, mu, sigma = peak["amplitude"], peak["center"], peak["sigma"]
            spectral_line += A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        results.append((i, j, continuum, spectral_line))

    return results
