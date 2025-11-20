import numpy as np

from surfh.Signalprocessing.model import gaussian


# =========================================================
# --- Line Extraction (1) ---
# =========================================================
def build_line_spectrum(baseline_subtrated, clean_peaks, scale=15):
    """Extract line spectrum using real data around fitted peaks."""
    line_spectrum = np.zeros_like(baseline_subtrated)
    for peak in clean_peaks:
        mu = int(round(peak['center']))
        sigma = peak['sigma']
        win_size = max(3, int(scale * sigma))
        start, end = max(0, mu - win_size // 2), min(len(baseline_subtrated), mu + win_size // 2 + 1)
        line_spectrum[start:end] = baseline_subtrated[start:end]
    return line_spectrum

# =========================================================
# --- Line Extraction (2) ---
# =========================================================
def build_line_spectrum_from_gaussians(fitted_peaks, n_points):
    """
    Reconstruct line spectrum using Gaussian parameters.
    
    Parameters
    ----------
    fitted_peaks : list of dict
        Each dict contains 'amplitude', 'center', 'sigma'.
    n_points : int
        Number of points in the original spectrum (for proper array size).
    
    Returns
    -------
    line_spectrum : np.ndarray
        Reconstructed spectrum from Gaussian fits.
    """
    line_spectrum = np.zeros(n_points)
    
    for peak in fitted_peaks:
        A = peak['amplitude']
        mu = peak['center']
        sigma = peak['sigma']
        
        # compute Gaussian over the full spectrum indices
        x = np.arange(n_points)
        line_spectrum += gaussian(x, A, mu, sigma)
    
    return line_spectrum
