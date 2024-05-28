import pytest
import numpy as np
from aljabr import LinOp, dottest
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.Models import slicer
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils
from astropy import units as u
from astropy.coordinates import Angle
from numpy.random import standard_normal as randn 

from surfh.Models import instru, slicer

from typing import List, Tuple
from numpy import ndarray as array

class RLTModel(LinOp):
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
        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        oshape = (self.instr.n_slit, len(self.instr.wavel_axis), self.slicer.npix_slit_alpha_width, self.slicer.npix_slit_beta_width)

        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)


    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step_degree)
    
    @property
    def beta_step(self) -> float:
        return self.beta_axis[1] - self.beta_axis[0]

    def _wpsf(self, length: int, step: float, wavel_axis: np.ndarray, instr: instru.IFU, wslice) -> np.ndarray:
        """Return spectral PSF"""
        # ∈ [0, β_s]
        beta_in_slit = np.arange(0, length) * step
        wpsf = instr.spectral_psf(
                        beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
                        wavel_axis[wslice],
                        arcsec2micron=instr.wavel_step / instr.det_pix_size,
                        type='mrs',
                    )  
        return wpsf

    def forward(self, maps: np.ndarray) -> np.ndarray:
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        allsliced = np.zeros(self.oshape)
        wpsf = self._wpsf(length=self.oshape[3],
                    step=self.beta_step,
                    wavel_axis=self.wavelength_axis,
                    instr=self.instr,
                    wslice=slice(0, len(self.wavelength_axis), None)
                    )
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(cube, slit_idx)
            blurred_sliced = jax_utils.wblur(sliced, wpsf)#cython_utils.wblur(sliced.astype(np.float64), (wpsf.conj()).astype(np.float64), 24)
            allsliced[slit_idx] = blurred_sliced
        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        wpsf = self._wpsf(length=self.oshape[3],
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        for slit_idx in range(self.instr.n_slit):
            blurred_t_sliced = jax_utils.wblur_t(inarray[slit_idx], wpsf.conj())#cython_utils.wblur_t(, 24)
            cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))

        maps = jax_utils.lmm_cube2maps(cube, self.templates).reshape(self.ishape)
        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)
    

