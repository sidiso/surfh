import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils


class T_spectro(LinOp):
        def __init__(
            self,
            maps,
            templates,
            wavelength_axis  
        ):
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.templates = templates # ex taille : [4, 307]
            self.maps = maps # ex taille : [4, 251, 251]

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            return jax_utils.lmm_maps2cube(maps, self.templates)
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            return jax_utils.lmm_cube2maps(cube, self.templates)