import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils, cython_utils
from surfh.Models import instru, slicer

from typing import List, Tuple
from numpy import ndarray as array


"""
Model : y = Rx

y : Hyperspectral cube of size (L', Nx, Ny)
R : Spectral blur operator
x : Hyperspectral cube of size (L, Nx, Ny)

# [λ', α, β] = ∑_λ arr[λ, α, β] wpsf[λ', λ, β]
# Σ_λ
"""
class spectroR(LinOp):
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


        ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))#maps.shape
        oshape = (len(self.instr.wavel_axis), len(alpha_axis), len(beta_axis))
        super().__init__(ishape=ishape, oshape=oshape)
        print(self.ishape, self.oshape)


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


    def forward(self, inarray: np.ndarray) -> np.ndarray:
        wpsf = self._wpsf(length=inarray.shape[1],
                step=self.step_degree,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        return cython_utils.wblur(inarray, wpsf, 1)
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        wpsf = self._wpsf(length=inarray.shape[1],
                step=self.step_degree,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        return cython_utils.wblur_t(inarray, wpsf.conj(), 1)

    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)