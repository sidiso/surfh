import pytest
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray as array
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils, cython_utils
from surfh.Models import instru, slicer



class spectroLST(LinOp):
    """
    Model : y = STa

    y : Hyperspectral Slits of size (NSlit, wl, slit_x, slit_y)
    L : Slicing operator
    S : Interpolation operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    def __init__(
        self,
        sotf: array,
        templates: array,
        alpha_axis: array,
        beta_axis: array,
        wavelength_axis: array,
        instr: instru.IFU,  
        step_Angle: float 
    ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.instr = instr
        self.step_Angle = step_Angle
        
        self.im_shape = (len(self.alpha_axis), len(self.beta_axis)) # ex taille : [251, 251]

        local_alpha_axis, local_beta_axis = self.instr.fov.local_coords(step_Angle.degree, 0, 0) # We don't use any margin
        
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis

        self.slicer = slicer.Slicer(self.instr, 
                                        wavelength_axis = self.wavelength_axis, 
                                        alpha_axis = self.alpha_axis, 
                                        beta_axis = self.beta_axis, 
                                        local_alpha_axis = self.local_alpha_axis, 
                                        local_beta_axis = self.local_beta_axis)
        
        self.local_shape =(len(self.wavelength_axis), len(local_alpha_axis), len(local_beta_axis))
        self.cube_shape = (len(self.wavelength_axis), len(self.alpha_axis), len(self.beta_axis))


        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        oshape = (self.instr.n_slit, len(self.wavelength_axis), self.slicer.npix_slit_alpha_width, self.slicer.npix_slit_beta_width)
        super().__init__(ishape=ishape, oshape=oshape)


    def forward(self, maps: np.ndarray) -> np.ndarray:
        # y = LSTa 
        cube = np.array(jax_utils.lmm_maps2cube(maps, self.templates)).astype(np.float64)

        local_alpha_coord, local_beta_coord = self.instr.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )
        optimized_local_coords = np.vstack(
                                        [
                                            local_alpha_coord.ravel(),
                                            local_beta_coord.ravel()
                                        ]
                                        ).T 
        gridded = np.array(cython_utils.interpn_cube2local(self.wavelength_axis, 
                                                   self.alpha_axis, 
                                                   self.beta_axis, 
                                                   cube, 
                                                   optimized_local_coords, 
                                                   self.local_shape))
        
        slits = np.zeros(self.oshape)
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(gridded, slit_idx)
            slits[slit_idx] = sliced

        return slits
    
    def adjoint(self, slits: np.ndarray) -> np.ndarray:
        # a = T^tS^tL^t y 

        # L^t
        cube = np.zeros(self.local_shape)
        for slit_idx in range(self.instr.n_slit):
            sliced = slits[slit_idx]
            cube += self.slicer.slicing_t(sliced, slit_idx, self.local_shape)

        alpha_coord, beta_coord = self.instr.fov.global2local(
                self.alpha_axis, self.beta_axis
                )
        
        optimized_global_coords = np.vstack(
            [
                alpha_coord.ravel(),
                beta_coord.ravel()
            ]
            ).T

        # S^t
        gridded_t = np.array(cython_utils.interpn_local2cube(self.wavelength_axis, 
                                                   self.local_alpha_axis.ravel(), 
                                                   self.local_beta_axis.ravel(), 
                                                   cube, 
                                                   optimized_global_coords, 
                                                   self.cube_shape))
        # T^t
        return jax_utils.lmm_cube2maps(gridded_t, self.templates)


    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)