import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils



class spectroCT(LinOp):
    """
    Model : y = CTa

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    def __init__(
        self,
        sotf,
        templates,
        alpha_axis,
        beta_axis,
        wavelength_axis  
    ):
        self.sotf = sotf
        self.templates = templates # ex taille : [4, 311]
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis # ex taille [311]
        
        self.im_shape = (len(self.alpha_axis), len(self.beta_axis)) # ex taille : [251, 251]

        ishape = (self.templates.shape[0], self.im_shape[0], self.im_shape[1]) #maps.shape
        oshape = (len(self.wavelength_axis), self.im_shape[0], self.im_shape[1])
        super().__init__(ishape=ishape, oshape=oshape)


    def forward(self, maps: np.ndarray) -> np.ndarray:
        # y = CTa 
        cube = jax_utils.lmm_maps2cube(maps, self.templates)
        return jax_utils.idft(jax_utils.dft(cube) * self.sotf, self.im_shape)
    
    def adjoint(self, cube: np.ndarray) -> np.ndarray:
        # a = T^tC^t y 
        conv_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf.conj(), self.im_shape)
        return jax_utils.lmm_cube2maps(conv_cube, self.templates)

    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)