import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils
from astropy import units as u
from astropy.coordinates import Angle
from numpy.random import standard_normal as randn 

from surfh.Models import instru, slicer, newslicer

from typing import List, Tuple
from numpy import ndarray as array


"""
Model : y = STx

y : Hyperspectral cube of size (L, Sx, Sy)
S : Slicing operation
T : LMM transformation operator
x : Hyperspectral cube of size (4, Nx, Ny)
"""
class SlicerModel(LinOp):
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
        
        self.newslicer = newslicer.newSlicer(self.instr, 
                                    wavelength_axis = self.wavelength_axis, 
                                    alpha_axis = self.alpha_axis, 
                                    beta_axis = self.beta_axis, 
                                    local_alpha_axis = self.local_alpha_axis, 
                                    local_beta_axis = self.local_beta_axis)

        # Templates (4, Nx, Ny)
        ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        oshape = (self.instr.n_slit, len(self.wavelength_axis), self.newslicer.npix_slit_alpha_width, self.slicer.npix_beta_slit)


        super().__init__(ishape=ishape, oshape=oshape)


    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step_degree)

    def forward(self, cube: array) -> array:
        allsliced = np.zeros(self.oshape)
        for slit_idx in range(self.instr.n_slit):
            sliced = self.newslicer.slicing(cube, slit_idx)
            allsliced[slit_idx] = sliced
        return allsliced


    def adjoint(self, slices: array) -> array:
        cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        for slit_idx in range(self.instr.n_slit):
            sliced = slices[slit_idx]
            cube += self.newslicer.slicing_t(sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
        return cube


maps = global_variable_testing.maps
templates = global_variable_testing.templates
im_shape = global_variable_testing.im_shape
wavelength_axis = global_variable_testing.wavelength_axis
sotf = global_variable_testing.sotf

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

cube_origin_alpha = 0
cube_origin_beta = 0
cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
cube_alpha_axis -= np.mean(cube_alpha_axis)
cube_beta_axis -= np.mean(cube_beta_axis)
cube_alpha_axis += cube_origin_alpha
cube_beta_axis += cube_origin_beta

alpha_width_fov_rchan = (cube_alpha_axis[-1] - cube_alpha_axis[0])# - step_Angle.degree*10
beta_width_fov_rchan = (cube_beta_axis[-1] - cube_beta_axis[0])# - step_Angle.degree*10

# Def random instru spec.
rchan = instru.IFU(
    fov=instru.FOV(alpha_width_fov_rchan, beta_width_fov_rchan, origin=instru.Coord(0, 0), angle=0),
    det_pix_size=0.196,
    n_slit=34,
    w_blur=None,
    pce=None,
    wavel_axis=wavelength_axis,
    name="2A",
)

sliceModel = SlicerModel(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)
cube = jax_utils.lmm_maps2cube(maps, templates)
slices = sliceModel.forward(cube)
ncube = sliceModel.adjoint(slices)
