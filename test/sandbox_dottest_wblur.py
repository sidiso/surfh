import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing
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


# def _wpsf(length: int, step: float, wavel_axis: np.ndarray, instr: instru.IFU, wslice) -> np.ndarray:
#         """Return spectral PSF"""
#         # ∈ [0, β_s]
#         beta_in_slit = np.arange(0, length) * step
#         wpsf = instr.spectral_psf(
#                         beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
#                         wavel_axis[wslice],
#                         arcsec2micron=instr.wavel_step / instr.det_pix_size,
#                         type='mrs',
#                     )  
#         return wpsf


# templates = global_variable_testing.templates
# instr_wavelength_axis = global_variable_testing.wavelength_axis
# n_lamnda = 200
# cube_walength_step = (instr_wavelength_axis[1]-instr_wavelength_axis[0])*2 # Set arbitrary wavel_axis 
# wavelength_axis = np.arange(instr_wavelength_axis[0], instr_wavelength_axis[-1], cube_walength_step)
# n_lamnda = len(wavelength_axis)

# im_shape = global_variable_testing.im_shape
# local_shape = (im_shape[0]-100, im_shape[1]-100)

# cube = np.random.random((n_lamnda, im_shape[0], im_shape[1])) # Make smaller cube for easier memory computation
# out_cube = np.zeros_like(cube)

# step = 0.025 # arcsec
# step_Angle = Angle(step, u.arcsec)


# grating_resolution = global_variable_testing.grating_resolution
# spec_blur = global_variable_testing.spec_blur

# # Def Channel spec.
# rchan = instru.IFU(
#     fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
#     det_pix_size=0.196,
#     n_slit=17,
#     w_blur=spec_blur,
#     pce=None,
#     wavel_axis=instr_wavelength_axis,
#     name="2A",
# )

# wpsf = _wpsf(length=cube.shape[1],
#                 step=step_Angle.degree,
#                 wavel_axis=wavelength_axis,
#                 instr=rchan,
#                 wslice=slice(0, n_lamnda, None)
#                 )



# python_wblurred_cube = python_utils.wblur(cube, wpsf)
# cython_wblurred_cube = cython_utils.wblur(cube, wpsf, 1)
# np.allclose(python_wblurred_cube, cython_wblurred_cube)

# t_python_wblurred_cube = python_utils.wblur_t(python_wblurred_cube, wpsf.conj())
# t_cython_wblurred_cube = cython_utils.wblur_t(python_wblurred_cube, wpsf.conj(), 1)
# np.any(t_python_wblurred_cube== t_cython_wblurred_cube)




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
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(cube, slit_idx)
        
            wpsf = self._wpsf(length=sliced.shape[2],
                    step=self.beta_step,
                    wavel_axis=self.wavelength_axis,
                    instr=self.instr,
                    wslice=slice(0, len(self.wavelength_axis), None)
                    )
            blurred_sliced = cython_utils.wblur(sliced.astype(np.float64), (wpsf.conj()).astype(np.float64), 1)
            allsliced[slit_idx] = blurred_sliced
        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        for slit_idx in range(self.instr.n_slit):
            wpsf = self._wpsf(length=inarray.shape[2],
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
            blurred_t_sliced = cython_utils.wblur_t(inarray[slit_idx], wpsf.conj(), 1)
            cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))

        maps = jax_utils.lmm_cube2maps(cube, self.templates).reshape(self.ishape)
        return maps
    

im_slice = slice(0,150,None)
templates = global_variable_testing.templates
maps = global_variable_testing.maps
maps = maps[:,im_slice, im_slice]

instr_wavelength_axis = global_variable_testing.chan_wavelength_axis
wavelength_axis = global_variable_testing.wavelength_axis
n_lamnda = len(wavelength_axis)
sotf = global_variable_testing.sotf

im_shape = global_variable_testing.im_shape
im_shape = (im_slice.stop, im_slice.stop)

cube = np.random.random((n_lamnda, im_shape[0], im_shape[1])) # Make smaller cube for easier memory computation
out_cube = np.zeros_like(cube)

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

cube_origin_alpha = 0
cube_origin_beta = 0
cube_alpha_axis = np.arange(cube.shape[1]).astype(np.float64)* step_Angle.degree
cube_beta_axis = np.arange(cube.shape[2]).astype(np.float64)* step_Angle.degree
cube_alpha_axis -= np.mean(cube_alpha_axis)
cube_beta_axis -= np.mean(cube_beta_axis)
cube_alpha_axis += cube_origin_alpha
cube_beta_axis += cube_origin_beta


grating_resolution = global_variable_testing.grating_resolution
spec_blur = global_variable_testing.spec_blur

# Def Channel spec.
rchan = instru.IFU(
    fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur,
    pce=None,
    wavel_axis=instr_wavelength_axis,
    name="2A",
)

rltModel = RLTModel(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)
data = rltModel.forward(maps)
proj = rltModel.adjoint(data)