import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from surfh.Signalprocessing.model import gaussian
from surfh.Signalprocessing.utilities import mad_std


# =========================================================
# --- Peak Detection & Fitting ---
# =========================================================
def detect_and_fit_peaks(baseline_subtrated, mad, sigma=5, distance=5, wavelength=None):
    """Detect peaks and fit Gaussians."""
    peaks, properties = find_peaks(baseline_subtrated, height=sigma * mad, distance=distance)
    print(f"Properties of detected peaks: {properties}")
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

            # --- Build line spectrum from data around fitted peak ---
            mu = int(round(popt[1]))
            sigma = popt[2]
            win_size = max(3, int(scale * sigma))
            l_start, l_end = max(0, mu - win_size // 2), min(len(baseline_subtracted), mu + win_size // 2 + 1)

            # --- Mirror neighboring values for continuum replacement ---
            masked_len = l_end - l_start
            if masked_len <= 0:
                continue

            # Left and right neighbors (extend if near edges)
            left_vals = continuum[max(0, l_start - masked_len):l_start]
            right_vals = continuum[l_end:min(len(continuum), l_end + masked_len)]

            if len(left_vals) == 0 or len(right_vals) == 0:
                continue

            # If not enough points, pad by repeating edge values
            if len(left_vals) < masked_len:
                left_vals = np.pad(left_vals, (masked_len - len(left_vals), 0), mode='edge')
            if len(right_vals) < masked_len:
                right_vals = np.pad(right_vals, (0, masked_len - len(right_vals)), mode='edge')

            # Mirror and average
            continuum[l_start:l_end] = (left_vals[::-1] + right_vals) / 2

            fitted_peaks.append({
                'peak_index': pk,
                'amplitude': popt[0],
                'center': popt[1],
                'sigma': popt[2]
            })

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
