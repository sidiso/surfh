import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



def baseline_als(y, lam=5e2, p=0.5, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    
    Args:
        y (np.ndarray): Input signal (1D array).
        lam (float): Smoothness parameter.
        p (float): Asymmetry parameter.
        niter (int): Number of iterations.
    Returns:
        np.ndarray: Estimated baseline.
    """
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
    """
    Iteratively estimate and subtract baseline.
    
    Args:
        y (np.ndarray): Input signal (1D array).
        lam (float): Smoothness parameter for baseline_als.
        p (float): Asymmetry parameter for baseline_als.
        niter (int): Number of iterations for baseline_als.
        ncycles (int): Number of cycles for iterative removal.
        sigma (float): Sigma threshold for outlier detection.
    Returns:
        np.ndarray: Final estimated baseline.
    """
    flux = np.array(y, dtype=float)
    for _ in range(ncycles):
        baseline = baseline_als(flux, lam, p, niter)
        baseline_subtrated = flux - baseline
        std = np.nanstd(baseline_subtrated)
        flux[baseline_subtrated > sigma * std] = np.nan
    return baseline_als(flux, lam, p, niter)
