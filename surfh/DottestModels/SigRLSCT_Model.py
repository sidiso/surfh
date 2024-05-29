import pytest
import numpy as np
from aljabr import LinOp, dottest
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.Models import slicer
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils, nearest_neighbor_interpolation
from astropy import units as u
from astropy.coordinates import Angle
from numpy.random import standard_normal as randn 
from math import ceil


from surfh.Models import instru, slicer

from typing import List, Tuple
from numpy import ndarray as array


"""
Model : y = SigRLSCTx

y : Hyperspectral slices of size (Nslices, L, Sx)
Sig : Beta subsampling operator
R : Spectral blur operator
L : Slicing operator
S : Interpolation operator
C : Spatial convolution operator
T : LMM operator
x : Hyperspectral cube of size (4, Nx, Ny)
"""
class spectroSigRLSCT(LinOp):
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
        oshape = (self.instr.n_slit, len(self.instr.wavel_axis), ceil(self.slicer.npix_slit_alpha_width / self.srf))

        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        self.local_cube_shape = (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr =python_utils.udft.ir2fr(np.ones((self.srf, 1)), self.local_cube_shape[1:])[np.newaxis, ...]


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
        # T
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))
        allsliced = np.zeros(self.oshape)
        wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                    step=self.beta_step,
                    wavel_axis=self.wavelength_axis,
                    instr=self.instr,
                    wslice=slice(0, len(self.wavelength_axis), None)
                    )
        
        local_alpha_coord, local_beta_coord = self.instr.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )
        optimized_local_coords = np.vstack(
                                            [
                                                local_alpha_coord.ravel(),
                                                local_beta_coord.ravel()
                                            ]
                                            ).T 
        
        
        # S
        gridded = cython_utils.interpn_cube2local(self.wavelength_axis.astype(np.float64), 
                                                   self.alpha_axis, 
                                                   self.beta_axis, 
                                                   np.array(blurred_cube).astype(np.float64), 
                                                   optimized_local_coords, 
                                                   (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))
        sum_cube = jax_utils.idft(
            jax_utils.dft(gridded) * self._otf_sr,
            self.local_cube_shape[1:],
        )
        for slit_idx in range(self.instr.n_slit):
            #L
            sliced = self.slicer.slicing(sum_cube, slit_idx)
            # SigR
            blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, wpsf)
            allsliced[slit_idx] = blurred_sliced_subsampled[:, : self.oshape[2] * self.srf : self.srf]
        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
        wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        for slit_idx in range(self.instr.n_slit):
            oversampled_sliced = np.repeat(
                    np.expand_dims(
                        inarray[slit_idx],
                        axis=2,
                    ),
                    self.slicer.npix_slit_beta_width,
                    axis=2,
                )
            blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
            blurred_t_sliced[:,: self.oshape[2] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced, wpsf.conj())
            local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))


        sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj(), self.local_cube_shape[1:])

        alpha_coord, beta_coord = self.instr.fov.global2local(
                    self.alpha_axis, self.beta_axis
                    )
            
        optimized_global_coords = np.vstack(
                [
                    alpha_coord.ravel(),
                    beta_coord.ravel()
                ]
                ).T

        degridded = cython_utils.interpn_local2cube(self.wavelength_axis, 
                                                   self.local_alpha_axis.ravel(), 
                                                   self.local_beta_axis.ravel(), 
                                                   np.array(sum_t_cube, dtype=np.float64), 
                                                   optimized_global_coords, 
                                                   self.cube_shape)
        
        blurred_t_cube = jax_utils.idft(jax_utils.dft(degridded) * self.sotf.conj(), (self.ishape[1], self.ishape[2]))
        maps = jax_utils.lmm_cube2maps(degridded, self.templates).reshape(self.ishape)
        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)
    



"""
Model : y = SigRLSCTx

y : Hyperspectral slices of size (Nslices, L, Sx)
Sig : Beta subsampling operator
R : Spectral blur operator
L : Slicing operator
S : Interpolation operator using nearest neighbor approach
C : Spatial convolution operator
T : LMM operator
x : Hyperspectral cube of size (4, Nx, Ny)
"""
class spectroSigRLSCT_NN(LinOp):
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
        oshape = (self.instr.n_slit, len(self.instr.wavel_axis), ceil(self.slicer.npix_slit_alpha_width / self.srf))

        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        self.local_cube_shape = (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr =python_utils.udft.ir2fr(np.ones((self.srf, 1)), self.local_cube_shape[1:])[np.newaxis, ...]
        self.precompute_mask()


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

    def precompute_mask(self):
        rin = np.ones(self.ishape)
        cube_rin = np.array(jax_utils.lmm_maps2cube(rin, self.templates)).astype(np.float64)

        local_alpha_coord, local_beta_coord = self.instr.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )
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
        # T
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))*self.nmask
        allsliced = np.zeros(self.oshape)
        wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                    step=self.beta_step,
                    wavel_axis=self.wavelength_axis,
                    instr=self.instr,
                    wslice=slice(0, len(self.wavelength_axis), None)
                    )
        
        local_alpha_coord, local_beta_coord = self.instr.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )

        test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
        test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis)) 

        # S
        wavel_idx = np.arange(len(self.wavelength_axis))
        indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), blurred_cube[0].ravel(), (local_alpha_coord, local_beta_coord))
        wavel_indexes = np.tile(indexes, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*blurred_cube[0].size )
        gridded = blurred_cube.ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])

        sum_cube = jax_utils.idft(
            jax_utils.dft(gridded) * self._otf_sr,
            self.local_cube_shape[1:],
        )

        for slit_idx in range(self.instr.n_slit):
            #L
            sliced = self.slicer.slicing(sum_cube, slit_idx)
            # SigR
            blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, wpsf)
            allsliced[slit_idx] = blurred_sliced_subsampled[:, : self.oshape[2] * self.srf : self.srf]
        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        local_cube = np.zeros((len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))
        wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        for slit_idx in range(self.instr.n_slit):
            oversampled_sliced = np.repeat(
                    np.expand_dims(
                        inarray[slit_idx],
                        axis=2,
                    ),
                    self.slicer.npix_slit_beta_width,
                    axis=2,
                )
            blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
            blurred_t_sliced[:,: self.oshape[2] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced, wpsf.conj())
            local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))

        sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj(), self.local_cube_shape[1:])

        local_alpha_coord, local_beta_coord = self.instr.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )

        test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
        test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis))  

        wavel_idx = np.arange(len(self.wavelength_axis))

        indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), sum_t_cube[0].ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
        wavel_indexes_t = np.tile(indexes_t, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes_t)) + ((wavel_idx[...,np.newaxis])*local_alpha_coord.shape[0]* local_alpha_coord.shape[1] )
        degridded = sum_t_cube.ravel()[wavel_indexes_t].reshape(self.cube_shape)


        blurred_t_cube = jax_utils.idft(jax_utils.dft(degridded*self.nmask) * self.sotf.conj(), (self.ishape[1], self.ishape[2]))
        maps = jax_utils.lmm_cube2maps(blurred_t_cube, self.templates).reshape(self.ishape)
        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)