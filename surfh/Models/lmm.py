import numpy as np
from surfh.ToolsDir import jax_utils, python_utils

class LinearMixingModel:
    """
    Linear Mixing Model (LMM) for hyperspectral data.
    """

    @staticmethod
    def mapsToCube(maps, templates):
        """
        Convert abundance maps and spectral templates to a hyperspectral cube.

        Parameters:
        maps (np.ndarray): Abundance maps of shape (n_components, height, width).
        templates (np.ndarray): Spectral templates of shape (n_components, n_wavelengths).

        Returns:
        np.ndarray: Hyperspectral cube of shape (n_wavelengths, height, width).
        """
        return  jax_utils.lmm_maps2cube(maps, templates)