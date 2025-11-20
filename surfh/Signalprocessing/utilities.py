import numpy as np

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
