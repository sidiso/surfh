import pytest
import numpy as np
from numpy import ndarray as array
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils, cython_utils, nearest_neighbor_interpolation
from surfh.Models import instru
from math import ceil




class spectroST(LinOp):
    """
    Model : y = STa

    y : Hyperspectral cube of size (L, Small_x, Small_y)
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
        step_Angle: float ,
        pointings: instru.CoordList
    ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.instr = instr
        self.step_Angle = step_Angle
        
        self.im_shape = (len(self.alpha_axis), len(self.beta_axis)) # ex taille : [251, 251]

        local_alpha_axis, local_beta_axis = self.instr.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
        
        self.pointings = pointings.pix(step_Angle.degree)

        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))

        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        oshape = (len(self.pointings), len(self.wavelength_axis), len(local_alpha_axis), len(local_beta_axis))
        super().__init__(ishape=ishape, oshape=oshape)


    def forward(self, maps: np.ndarray) -> np.ndarray:
        # y = STa 
        cube = np.array(jax_utils.lmm_maps2cube(maps, self.templates)).astype(np.float64)
        output = np.zeros(self.oshape)

        for p_idx, pointing in enumerate(self.pointings):
            local_alpha_coord, local_beta_coord = (self.instr.fov + pointing).local2global(
                                                            self.local_alpha_axis, self.local_beta_axis
                                                            )
            optimized_local_coords = np.vstack(
                                            [
                                                local_alpha_coord.ravel(),
                                                local_beta_coord.ravel()
                                            ]
                                            ).T 
            output[p_idx] = np.array(cython_utils.interpn_cube2local(self.wavelength_axis, 
                                                   self.alpha_axis, 
                                                   self.beta_axis, 
                                                   cube, 
                                                   optimized_local_coords, 
                                                   self.oshape[1:]))
        
        return output
    
    def adjoint(self, cube: np.ndarray) -> np.ndarray:
        # a = T^tS^t y 
        out = np.zeros(self.cube_shape)
        for p_idx, pointing in enumerate(self.pointings):
            alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
                    self.alpha_axis, self.beta_axis
                    )
            
            optimized_global_coords = np.vstack(
                [
                    alpha_coord.ravel(),
                    beta_coord.ravel()
                ]
                ).T

            gridded_t = np.array(cython_utils.interpn_local2cube(self.wavelength_axis, 
                                                    self.local_alpha_axis.ravel(), 
                                                    self.local_beta_axis.ravel(), 
                                                    cube[p_idx], 
                                                    optimized_global_coords, 
                                                    self.cube_shape))
            out += gridded_t
        return jax_utils.lmm_cube2maps(out, self.templates)


    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)