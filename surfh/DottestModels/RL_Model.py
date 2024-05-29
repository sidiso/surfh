import pytest
import numpy as np
from aljabr import LinOp, dottest

from surfh.ToolsDir import jax_utils, python_utils, cython_utils
from surfh.Models import instru, slicer

from typing import List, Tuple
from math import ceil
from numpy import ndarray as array


"""
Model : y = RLTx

y : Hyperspectral slices of size (Nslices, L', Sx, Sy)
R : Spectral blur operator
L : Slicing operation
x : Hyperspectral cube of size (4, Nx, Ny)
"""
class spectroRL(LinOp):
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
        ishape = (len(wavelength_axis), len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        oshape = (self.instr.n_slit, len(self.instr.wavel_axis), ceil(self.slicer.npix_slit_alpha_width / self.srf), self.slicer.npix_slit_beta_width)
        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))
        super().__init__(ishape=ishape, oshape=oshape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr =python_utils.udft.ir2fr(np.ones((self.srf, 1)), self.ishape[1:])[np.newaxis, ...]


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

    def forward(self, cube: np.ndarray) -> np.ndarray:
        allsliced = np.zeros(self.oshape)

        wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        sum_cube = jax_utils.idft(
            jax_utils.dft(cube) * self._otf_sr,
            self.imshape,
        )

        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(sum_cube, slit_idx)
            blurred_sliced = jax_utils.wblur(sliced, wpsf)
            allsliced[slit_idx] = blurred_sliced[:, : self.oshape[2] * self.srf : self.srf,:]

        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        for slit_idx in range(self.instr.n_slit):
            blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
            blurred_t_sliced[:,: self.oshape[2] * self.srf : self.srf,:] = jax_utils.wblur_t(inarray[slit_idx], wpsf.conj())
            cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))

        sum_t_cube = jax_utils.idft(jax_utils.dft(cube) * self._otf_sr.conj(), self.ishape[1:])

        return sum_t_cube