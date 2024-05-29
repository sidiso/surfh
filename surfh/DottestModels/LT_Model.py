import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils
from surfh.Models import instru, slicer
from math import ceil


from typing import List, Tuple
from numpy import ndarray as array
from math import ceil
import matplotlib.pyplot as plt


"""
Model : y = LTx

y : Hyperspectral slices of size (Nslices, L', Sx, Sy)
L : Slicing operation
T : LMM transformation operator
x : Hyperspectral cube of size (4, Nx, Ny)
"""
class spectroLT(LinOp):
    def __init__(
                self,
                sotf: array,
                templates: array,
                alpha_axis: array,
                beta_axis: array,
                wavelength_axis: array,
                instr: instru.IFU,  
                step_degree: float
                ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.instr = instr
        self.step_degree = step_degree

        local_alpha_axis, local_beta_axis = self.instr.fov.local_coords(step_degree, 0, 0)#5* step_degree, 5* step_degree)
        
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis

        self.slicer = slicer.Slicer(self.instr, 
                                    wavelength_axis = self.wavelength_axis, 
                                    alpha_axis = self.alpha_axis, 
                                    beta_axis = self.beta_axis, 
                                    local_alpha_axis = self.local_alpha_axis, 
                                    local_beta_axis = self.local_beta_axis)
        
        # Super resolution factor (in alpha dim)
        self.srf = instru.get_srf(
            [self.instr.det_pix_size],
            self.step_degree*3600,
        )[0]
        print(f'Super Resolution factor is set to {self.srf}')

        

        # Templates (4, Nx, Ny)
        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        oshape = (self.instr.n_slit, len(self.wavelength_axis), ceil(self.slicer.npix_slit_alpha_width / self.srf), self.slicer.npix_slit_beta_width)  # self.n_alpha // self.srf,

        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))
        super().__init__(ishape=ishape, oshape=oshape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr =python_utils.udft.ir2fr(np.ones((self.srf, 1)), self.ishape[1:])[np.newaxis, ...]


    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step_degree)

    def forward(self, maps: array) -> array:
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)

        sum_cube = jax_utils.idft(
            jax_utils.dft(cube) * self._otf_sr,
            self.imshape,
        )
        allsliced = np.zeros(self.oshape)
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(sum_cube, slit_idx)
            allsliced[slit_idx] = sliced[:, : self.oshape[2] * self.srf : self.srf,:]
        return allsliced


    def adjoint(self, slices: array) -> array:
        cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        
        for slit_idx in range(self.instr.n_slit):
            sliced = np.zeros(self.slicer.get_slit_shape())
            sliced[:,: self.oshape[2] * self.srf : self.srf,:] = slices[slit_idx]
            cube += self.slicer.slicing_t(sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))

        sum_t_cube = jax_utils.idft(jax_utils.dft(cube) * self._otf_sr.conj(), self.ishape[1:])
        maps = jax_utils.lmm_cube2maps(sum_t_cube, self.templates).reshape(self.ishape)
        return maps
    

    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)