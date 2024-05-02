import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing

from surfh.ToolsDir import jax_utils, python_utils


def test_LMM_python_dottest():
    """
    Model : y = Ta

    y : Hyperspectral cube of size (L, Nx, Ny)
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    class LMM(LinOp):
        def __init__(
            self,
            maps,
            templates,
            wavelength_axis  
        ):
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.templates = templates # ex taille : [4, 307]
            self.maps = maps # ex taille : [4, 251, 251]

            ishape = (4,251,251)#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            maps = maps.reshape(self.ishape)
            return python_utils.lmm_maps2cube(maps, self.templates).reshape(self.oshape)
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            cube = cube.reshape(self.oshape)
            return python_utils.lmm_cube2maps(cube, self.templates).reshape(self.ishape)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    lmm_class = LMM(maps, templates, wavelength_axis)

    assert dottest(lmm_class)



