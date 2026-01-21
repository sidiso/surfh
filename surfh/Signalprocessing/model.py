import numpy as np
from scipy.special import voigt_profile

def gaussian(x, A, mu, sigma):
    """
    Gaussian function with equation : A * exp(-(x - mu)^2 / (2 * sigma^2)) 

    Args:
        x (np.ndarray): Input array.
        A (float): Amplitude of the Gaussian.
        mu (float): Mean (center) of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: Computed Gaussian values at x.
    """
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def voigt(x, A, mu, sigma, gamma):
    """
    Voigt profile function.
    Args:
        x (np.ndarray): Input array.
        A (float): Amplitude of the Voigt profile.
        mu (float): Center of the Voigt profile.
        sigma (float): Gaussian component standard deviation.
        gamma (float): Lorentzian component half-width at half-maximum. 
    Returns:
        np.ndarray: Computed Voigt profile values at x.
    """
    return A * voigt_profile(x - mu, sigma, gamma)

