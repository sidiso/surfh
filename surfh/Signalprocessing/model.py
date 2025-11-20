import numpy as np
from scipy.special import voigt_profile

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def voigt(x, A, mu, sigma, gamma):
    return A * voigt_profile(x - mu, sigma, gamma)

