import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils

class spectroC(LinOp):
    def __init__(
        self,
        sotf,
        maps,
        templates,
        wavelength_axis  
    ):
        self.wavelength_axis = wavelength_axis # ex taille [307]
        self.sotf = sotf
        self.maps = maps

        ishape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])#maps.shape
        oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
        super().__init__(ishape=ishape, oshape=oshape)
        print(self.ishape, self.oshape)


    def forward(self, inarray: np.ndarray) -> np.ndarray:
        return jax_utils.idft(jax_utils.dft(inarray) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        return jax_utils.idft(jax_utils.dft(inarray) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))