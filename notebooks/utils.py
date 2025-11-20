import numpy as np
import matplotlib.pyplot as plt
from rich import print
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

# =========================================================
# --- Utilities ---
# =========================================================
def mad_std(x):
    """Robust standard deviation from Median Absolute Deviation (MAD)."""
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))

def sliding_mad(x):
    """Compute sliding MAD-based std over a 1D array."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    window = max(3, n // 10)
    if window % 2 == 0:
        window += 1
    half = window // 2
    result = np.zeros(n)
    for i in range(n):
        start, end = max(0, i - half), min(n, i + half + 1)
        result[i] = mad_std(x[start:end])
    return result

# =========================================================
# --- Models ---
# =========================================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def voigt(x, A, mu, sigma, gamma):
    return A * voigt_profile(x - mu, sigma, gamma)

# =========================================================
# --- Baseline Removal ---
# =========================================================
def baseline_als(y, lam=5e2, p=0.5, niter=10):
    """Asymmetric Least Squares baseline correction."""
    y = np.asarray(y, dtype=float)
    isnan = np.isnan(y)
    if np.any(isnan):
        not_nan = ~isnan
        y[isnan] = np.interp(np.flatnonzero(isnan),
                             np.flatnonzero(not_nan),
                             y[not_nan])
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def iterative_baseline_removal(y, lam=5e2, p=0.5, niter=10, ncycles=3, sigma=1.0):
    """Iteratively estimate and subtract baseline."""
    flux = np.array(y, dtype=float)
    for _ in range(ncycles):
        baseline = baseline_als(flux, lam, p, niter)
        baseline_subtrated = flux - baseline
        std = np.nanstd(baseline_subtrated)
        flux[baseline_subtrated > sigma * std] = np.nan
    return baseline_als(flux, lam, p, niter)

# =========================================================
# --- Peak Detection & Fitting ---
# =========================================================
def detect_and_fit_peaks(baseline_subtrated, mad, sigma=5, distance=5, wavelength=None):
    """Detect peaks and fit Gaussians."""
    peaks, _ = find_peaks(baseline_subtrated, height=sigma * mad, distance=distance)
    fitted_peaks = []
    remaining_peaks = peaks.copy()

    for pk in peaks:
        if pk not in remaining_peaks:
            continue
        mask_range = np.arange(pk - 5, pk + 6)
        remaining_peaks = np.setdiff1d(remaining_peaks, mask_range, assume_unique=True)

        start, end = max(0, pk - 10), min(len(baseline_subtrated), pk + 10)
        x_window = np.arange(start, end)
        y_window = baseline_subtrated[start:end]

        p0 = [y_window.max() - y_window.min(), pk, 1]
        try:
            popt, _ = curve_fit(gaussian, x_window, y_window, p0=p0)
            if wavelength is None:
                fitted_peaks.append({'peak_index': pk, 'amplitude': popt[0],
                                     'center': popt[1], 'sigma': popt[2]})
            else:
                i0 = int(np.floor(popt[1]))
                i1 = int(np.ceil(popt[1]))
                frac = popt[1] - i0
                wavelength_center = wavelength[i0] * (1 - frac) + wavelength[i1] * frac
                fitted_peaks.append({'peak_index': pk, 'amplitude': popt[0],
                                     'center': popt[1], 'wavel_center': wavelength_center, 'sigma': popt[2]})
        except RuntimeError:
            pass

    return np.array(fitted_peaks)

def fit_peaks_only(baseline_subtracted, input_spectrum, peak_indices, mean_sigma, std_sigma, window=10, scale=15):
    """
    Fit Gaussians to specified peak positions and build continuum axis
    using mirrored neighboring values for continuum replacement.
    """
    baseline_subtracted = np.asarray(baseline_subtracted, dtype=float)
    fitted_peaks = []
    continuum = input_spectrum.copy()

    for pk in peak_indices:
        start, end = max(0, pk - window), min(len(baseline_subtracted), pk + window + 1)
        x_window = np.arange(start, end)
        y_window = baseline_subtracted[start:end]

        p0 = [y_window.max() - y_window.min(), pk, mean_sigma]
        try:
            popt, _ = curve_fit(
                gaussian, x_window, y_window, p0=p0,
                bounds=([0, 0.9999 * pk, 0],
                        [np.inf, 1.0001 * pk, mean_sigma + 3 * std_sigma])
            )
            fitted_peaks.append({
                'peak_index': pk,
                'amplitude': popt[0],
                'center': popt[1],
                'sigma': popt[2]
            })

            # --- Build line spectrum from data around fitted peak ---
            mu = int(round(popt[1]))
            sigma = popt[2]
            win_size = max(3, int(scale * sigma))
            l_start, l_end = max(0, mu - win_size // 2), min(len(baseline_subtracted), mu + win_size // 2 + 1)

            # --- Mirror neighboring values for continuum replacement ---
            masked_len = l_end - l_start

            # Left and right neighbors (extend if near edges)
            left_vals = continuum[max(0, l_start - masked_len):l_start]
            right_vals = continuum[l_end:min(len(continuum), l_end + masked_len)]

            # If not enough points, pad by repeating edge values
            if len(left_vals) < masked_len:
                left_vals = np.pad(left_vals, (masked_len - len(left_vals), 0), mode='edge')
            if len(right_vals) < masked_len:
                right_vals = np.pad(right_vals, (0, masked_len - len(right_vals)), mode='edge')

            # Mirror and average
            continuum[l_start:l_end] = (left_vals[::-1] + right_vals) / 2

        except RuntimeError:
            continue
        
    return fitted_peaks, continuum


def filter_clean_peaks(fitted_peaks):
    """Keep only peaks with reasonable sigma based on bright subset."""
    amplitudes = np.array([p['amplitude'] for p in fitted_peaks])
    sigmas = np.array([p['sigma'] for p in fitted_peaks])
    n_keep = max(1, len(amplitudes) // 5)
    sorted_idx = np.argsort(amplitudes)[::-1]
    top_idx = sorted_idx[:n_keep]
    mean_sigma = np.mean(sigmas[top_idx])
    std_sigma = mad_std(sigmas[top_idx])
    clean_peaks = [p for p in fitted_peaks if p['sigma'] < mean_sigma + 3 * std_sigma]
    return clean_peaks, mean_sigma, std_sigma

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

# =========================================================
# --- Plotting ---
# =========================================================
def plot_results(wave, flux, baseline, line_spectrum, lines_gaussian):
    _, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

    # Top panel: continuum vs. baseline
    axs[0].plot(wave, baseline, label='Baseline (ALS)', color='black', lw=2, ls=":", alpha=0.5)
    axs[0].plot(wave, flux, label='Input Spectrum', color='black', alpha=0.75)
    axs[0].plot(wave, flux - line_spectrum, label='Separated Continuum', color='red')
    axs[0].legend()

    # --- Bottom panel: Gaussian fits over extracted lines ---
    axs[1].plot(wave, flux - baseline, color='black', label='Separated lines', alpha=0.75)
    axs[1].plot(wave, lines_gaussian, color='teal', label='Gaussians (best fit)', alpha=0.75)
    axs[1].plot(wave, flux - baseline - lines_gaussian - 5e2, color='crimson', label='Residuals (shifted)', alpha=0.75)
    axs[1].set_xlabel("Wavelength (microns)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# =========================================================
# --- Main ---
# =========================================================
def main():
    # Load data
    flux = np.load("/home/pdellova/mean_spectrum.npy")
    wave = np.load("/home/pdellova/wave.npy")

    # Baseline subtraction
    baseline = iterative_baseline_removal(flux, lam=1e3, ncycles=5, sigma=1.0)
    baseline_subtrated = flux - baseline
    mad = sliding_mad(baseline_subtrated)

    # Peak fitting
    fitted_peaks = detect_and_fit_peaks(baseline_subtrated, mad, sigma=5, distance=5)
    clean_peaks, mean_sigma, std_sigma = filter_clean_peaks(fitted_peaks)

    print(clean_peaks)
    print(f"Clean peaks: {len(clean_peaks)}")
    print(f"Mean sigma (brightest 20%): {mean_sigma:.3f}")
    print(f"Std sigma: {std_sigma:.3f}")

    # Line spectrum extraction (1) - using the Gaussian fits only to extract real data around the peaks
    lines_realdata = build_line_spectrum(baseline_subtrated, clean_peaks)

    # Line spectrum extraction (2) - using the Gaussian fits as actual values
    lines_gaussian = build_line_spectrum_from_gaussians(clean_peaks, len(wave))

    # Plot results
    plot_results(wave, flux, baseline, lines_realdata, lines_gaussian)

if __name__ == "__main__":
    main()
