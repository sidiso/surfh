import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils
from surfh.Models import instru, slicer

from typing import List, Tuple
from numpy import ndarray as array


"""
Model : y = LTx

y : Hyperspectral slices of size (Nslices, L', Sx, Sy)
L : Slicing operation
T : LMM transformation operator
x : Hyperspectral cube of size (4, Nx, Ny)
"""
class LT_spectro(LinOp):
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
        
        # Templates (4, Nx, Ny)
        ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        oshape = (self.instr.n_slit, len(self.wavelength_axis), self.slicer.npix_slit_alpha_width, self.slicer.npix_slit_beta_width)


        super().__init__(ishape=ishape, oshape=oshape)


    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step_degree)

    def forward(self, cube: array) -> array:
        allsliced = np.zeros(self.oshape)
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(cube, slit_idx)
            allsliced[slit_idx] = sliced
        return allsliced


    def adjoint(self, slices: array) -> array:
        cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        for slit_idx in range(self.instr.n_slit):
            sliced = slices[slit_idx]
            cube += self.slicer.slicing_t(sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        return cube