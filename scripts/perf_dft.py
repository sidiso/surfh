import numpy as np
import udft
import scipy as sp
from typing import List, Tuple

import pyfftw

def idft(inarray: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Apply the unitary inverse Discret Fourier Transform on last two axis.

    Parameters
    ----------
    inarray: array-like
      The array to transform
    shape: tuple(int, int)
      The output shape of the last two axis.

    Notes
    -----
    Use `scipy.fft.irfftn` with `workers=-1`.
    """
    return sp.fft.irfftn(
        inarray, s=shape, axes=range(-len(shape), 0), norm="ortho", workers=-1
    )

def dft(inarray: np.ndarray) -> np.ndarray:
    """Apply the unitary Discret Fourier Transform on last two axis.

    Parameters
    ----------
    inarray: array-like
      The array to transform

    Notes
    -----
    Use `scipy.fft.rfftn` with `workers=-1`.
    """
    return sp.fft.rfftn(inarray, axes=range(-2, 0), norm="ortho", workers=-1)


shape = (3182, 251, 251)

np.random.seed(999)
cube = np.random.random(shape)

# Aligned cube for fftw
a = pyfftw.empty_aligned(shape, dtype='float32', n=16)
a[:] = cube

b = pyfftw.interfaces.scipy_fft.rfftn(a)
