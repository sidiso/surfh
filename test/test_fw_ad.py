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

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            return python_utils.lmm_maps2cube(maps, self.templates)
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            return python_utils.lmm_cube2maps(cube, self.templates)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    lmm_class = LMM(maps, templates, wavelength_axis)

    assert dottest(lmm_class)

def test_LMM_jax_dottest():
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

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            maps = maps.reshape(self.ishape)
            return jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.oshape)
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            cube = cube.reshape(self.oshape)
            return jax_utils.lmm_cube2maps(cube, self.templates).reshape(self.ishape)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    lmm_class = LMM(maps, templates, wavelength_axis)

    assert dottest(lmm_class)



def test_spatial_conv_python_dottest():
    """
    Model : y = Cx

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    """
    class Conv(LinOp):
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
            return python_utils.idft(python_utils.dft(inarray) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, inarray: np.ndarray) -> np.ndarray:
            return python_utils.idft(python_utils.dft(inarray) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf
    conv_class = Conv(sotf, maps, templates, wavelength_axis)

    assert dottest(conv_class)


def test_spatial_conv_jax_dottest():
    """
    Model : y = Cx

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    """
    class Conv(LinOp):
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


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf
    conv_class = Conv(sotf, maps, templates, wavelength_axis)

    assert dottest(conv_class)


def test_LMMConv_python_dottest():
    """
    Model : y = CTa

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    class LMMConv(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.sotf = sotf
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.templates = templates # ex taille : [4, 307]
            self.maps = maps # ex taille : [4, 251, 251]

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            cube = python_utils.lmm_maps2cube(maps, self.templates)
            return python_utils.idft(python_utils.dft(cube) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            conv_cube = python_utils.idft(python_utils.dft(cube) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))
            return python_utils.lmm_cube2maps(conv_cube, self.templates)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    lmmConv_class = LMMConv(sotf, maps, templates, wavelength_axis)

    assert dottest(lmmConv_class)

def test_LMMConv_jax_dottest():
    """
    Model : y = CTa

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    class LMMConv(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.sotf = sotf
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.templates = templates # ex taille : [4, 307]
            self.maps = maps # ex taille : [4, 251, 251]

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            cube = jax_utils.lmm_maps2cube(maps, self.templates)
            return jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            conv_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))
            return jax_utils.lmm_cube2maps(conv_cube, self.templates)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    lmmConv_class = LMMConv(sotf, maps, templates, wavelength_axis)

    assert dottest(lmmConv_class)


def test_interpn_python_dottest():
    """
    Model : y = Sx

    y : Hyperspectral cube of size (L, Nx, Ny)
    S : Interpolation operator
    x : Hyperspectral cube of size (4, Nx, Ny)
    """
    class Interpn(LinOp):
        def __init__(
                    self,
                    sotf,
                    templates,
                    alpha_axis,
                    beta_axis,
                    wavelength_axis  
                    ):


            super().__init__(ishape=ishape, oshape=oshape)

        def forward(self, point: np.ndarray) -> np.ndarray:
            return super().forward(point)
        
        def adjoint(self, point: np.ndarray) -> np.ndarray:
            return super().adjoint(point)
