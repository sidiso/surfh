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
        step_Angle: float 
    ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.inst = instr
        self.step_Angle = step_Angle
        
        self.im_shape = (len(self.alpha_axis), len(self.beta_axis)) # ex taille : [251, 251]

        local_alpha_axis, local_beta_axis = self.inst.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
        
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))

        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        oshape = (len(self.wavelength_axis), len(local_alpha_axis), len(local_beta_axis))
        super().__init__(ishape=ishape, oshape=oshape)


    def forward(self, maps: np.ndarray) -> np.ndarray:
        # y = STa 
        cube = np.array(jax_utils.lmm_maps2cube(maps, self.templates)).astype(np.float64)

        local_alpha_coord, local_beta_coord = self.inst.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )
        optimized_local_coords = np.vstack(
                                        [
                                            local_alpha_coord.ravel(),
                                            local_beta_coord.ravel()
                                        ]
                                        ).T 
        return np.array(cython_utils.interpn_cube2local(self.wavelength_axis, 
                                                   self.alpha_axis, 
                                                   self.beta_axis, 
                                                   cube, 
                                                   optimized_local_coords, 
                                                   self.oshape))
    
    def adjoint(self, cube: np.ndarray) -> np.ndarray:
        # a = T^tS^t y 
        alpha_coord, beta_coord = self.inst.fov.global2local(
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
                                                   cube, 
                                                   optimized_global_coords, 
                                                   self.cube_shape))
        return jax_utils.lmm_cube2maps(gridded_t, self.templates)


    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)
    


class spectroSnearestT(LinOp):
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
        step_Angle: float 
    ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.inst = instr
        self.step_Angle = step_Angle
        
        self.im_shape = (len(self.alpha_axis), len(self.beta_axis)) # ex taille : [251, 251]

        local_alpha_axis, local_beta_axis = self.inst.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
        
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))

        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        oshape = (len(self.wavelength_axis), len(local_alpha_axis), len(local_beta_axis))
        super().__init__(ishape=ishape, oshape=oshape)

        self.precompute_mask()


    def precompute_mask(self):
        rin = np.ones(self.ishape)
        cube_rin = np.array(jax_utils.lmm_maps2cube(rin, self.templates)).astype(np.float64)

        local_alpha_coord, local_beta_coord = self.inst.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )
        optimized_local_coords = np.vstack(
                                        [
                                            local_alpha_coord.ravel(),
                                            local_beta_coord.ravel()
                                        ]
                                        ).T
        test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
        test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis))    
        
        wavel_idx = np.arange(len(self.wavelength_axis))

        indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube_rin[0].ravel(), (local_alpha_coord, local_beta_coord))
        wavel_indexes = np.tile(indexes, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*cube_rin[0].size )
        gridata = cube_rin.ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])

        indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata[0].ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
        wavel_indexes_t = np.tile(indexes_t, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes_t)) + ((wavel_idx[...,np.newaxis])*local_alpha_coord.shape[0]* local_alpha_coord.shape[1] )
        degridata = gridata.ravel()[wavel_indexes_t].reshape(len(wavel_idx), 251,251)

        mask = np.zeros_like(cube_rin[0])
        mask.ravel()[indexes] = 1
        nmask = np.zeros_like(mask)
        for i in range(1,cube_rin.shape[1]-1):
            for j in range(1, cube_rin.shape[2]-1):
                if mask[i,j] == 1:
                    nmask[i,j] = 1
                else:
                    if mask[i-1, j-1] == 1 or mask[i, j-1] == 1 \
                        or mask[i+1, j-1] == 1 or mask[i-1, j] == 1\
                        or mask[i-1, j+1] == 1 or mask[i+1, j+1] == 1\
                        or mask[i, j+1] == 1 or mask[i+1, j] == 1:
                        nmask[i,j] = 1        
        self.nmask = nmask

    def forward(self, maps: np.ndarray) -> np.ndarray:
        # y = STa 
        cube = np.array(jax_utils.lmm_maps2cube(maps, self.templates)).astype(np.float64) * self.nmask

        local_alpha_coord, local_beta_coord = self.inst.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )

        test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
        test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis))    
        
        wavel_idx = np.arange(len(self.wavelength_axis))

        indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube[0].ravel(), (local_alpha_coord, local_beta_coord))
        wavel_indexes = np.tile(indexes, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*cube[0].size )
        gridata = cube.ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])
        return gridata
    
    def adjoint(self, cube: np.ndarray) -> np.ndarray:
        # a = T^tS^t y 
        local_alpha_coord, local_beta_coord = self.inst.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )

        test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
        test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis))  

        wavel_idx = np.arange(len(self.wavelength_axis))

        indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), cube[0].ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
        wavel_indexes_t = np.tile(indexes_t, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes_t)) + ((wavel_idx[...,np.newaxis])*local_alpha_coord.shape[0]* local_alpha_coord.shape[1] )
        degridata = cube.ravel()[wavel_indexes_t].reshape(self.cube_shape)

        return jax_utils.lmm_cube2maps(degridata*self.nmask, self.templates)


    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)