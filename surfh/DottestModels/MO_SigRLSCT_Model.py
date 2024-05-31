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

from surfh.Models import instru, slicer
from math import ceil


from typing import List, Tuple
from numpy import ndarray as array


"""
Multi-Observation Single Channel for the spectro model :

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
                step_degree: float,
                pointings: instru.CoordList
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

        self.pointings = pointings.pix(self.step_degree)  

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
        oshape = (len(self.pointings), self.instr.n_slit, len(self.instr.wavel_axis), ceil(self.slicer.npix_slit_alpha_width / self.srf))

        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        self.local_cube_shape = (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr =python_utils.udft.ir2fr(np.ones((self.srf, 1)), self.local_cube_shape[1:])[np.newaxis, ...]
        self.wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )
        """
        decal = np.zeros(self.local_cube_shape[1:])
        dsi = int((self.srf-1)/2)
        dsj = 0 # int((self.n_pix_beta_slit -1) /2)
        decal[-dsi, -dsj] = np.sqrt(self.local_cube_shape[1]*self.local_cube_shape[2])
        decalf = jax_utils.dft(decal)
        sum_cube = jax_utils.idft(
                jax_utils.dft(gridded) * self._otf_sr * decalf,
                self.local_cube_shape[1:],
            )
        res = sum_cube[:, ::srf, ::self.n_pix_beta_slit]

        Adjoint :
        local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
        local_cube[:,::self.srf, ::self.n_pix_beta_slit] = someArray
        sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*decalf.conj(), self.local_cube_shape[1:])

        """

    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step_degree)
    
    @property
    def alpha_step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]
    
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

    def gridding(self, blurred_cube: array, pointing: instru.Coord) -> array:
        local_alpha_coord, local_beta_coord = (self.instr.fov + pointing).local2global(
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
        
        return gridded
    
    def gridding_t(self, local_cube: array, pointing: instru.Coord) -> array:
        alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
                self.alpha_axis, self.beta_axis
                )
        
        optimized_global_coords = np.vstack(
                [
                    alpha_coord.ravel(),
                    beta_coord.ravel()
                ]
                ).T

        global_cube = cython_utils.interpn_local2cube(self.wavelength_axis, 
                                                self.local_alpha_axis.ravel(), 
                                                self.local_beta_axis.ravel(), 
                                                np.array(local_cube, dtype=np.float64), 
                                                optimized_global_coords, 
                                                self.cube_shape)
        return global_cube

    
    def forward(self, maps: np.ndarray) -> np.ndarray:
        # T
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))
        allsliced = np.zeros(self.oshape)
        
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred_cube, pointing) 
            sum_cube = jax_utils.idft(
                jax_utils.dft(gridded) * self._otf_sr,
                self.local_cube_shape[1:],
            )
            for slit_idx in range(self.instr.n_slit):
                #L
                sliced = self.slicer.slicing(sum_cube, slit_idx)
                # SigR
                blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.wpsf)
                allsliced[p_idx, slit_idx] = blurred_sliced_subsampled[:, : self.oshape[3] * self.srf : self.srf]
        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        global_cube = np.zeros(self.cube_shape)
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            inarray[p_idx, slit_idx],
                            axis=2,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=2,
                    )
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
                blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced, self.wpsf.conj())
                local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj(), self.local_cube_shape[1:])

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing)
            global_cube += degridded

        blurred_t_cube = jax_utils.idft(jax_utils.dft(global_cube) * self.sotf.conj(), (self.ishape[1], self.ishape[2]))
        maps = jax_utils.lmm_cube2maps(blurred_t_cube, self.templates).reshape(self.ishape)
        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)

    def make_mask(self, maps):
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        allsliced = np.zeros(self.oshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(cube, pointing) 
            sum_cube = jax_utils.idft(
                jax_utils.dft(gridded) * self._otf_sr,
                self.local_cube_shape[1:],
            )
            for slit_idx in range(self.instr.n_slit):
                #L
                sliced = self.slicer.slicing(sum_cube, slit_idx)
                # SigR
                blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.wpsf)
                allsliced[p_idx, slit_idx] = blurred_sliced_subsampled[:, : self.oshape[3] * self.srf : self.srf]


        global_cube = np.zeros(self.cube_shape)
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            allsliced[p_idx, slit_idx],
                            axis=2,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=2,
                    )
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
                blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced, self.wpsf.conj())
                local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj(), self.local_cube_shape[1:])

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing)
            global_cube += degridded


        mask = np.ones_like(global_cube)
        zeros = np.where(global_cube == 0.)
        mask[zeros] = 0

        return mask
    


"""
Multi-Observation Single Channel for the spectro model :

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
class spectroSigRLSCT_corrected(LinOp):
    def __init__(
                self,
                sotf: array,
                templates: array,
                alpha_axis: array,
                beta_axis: array,
                wavelength_axis: array,
                instr: instru.IFU,  
                step_degree: float,
                pointings: instru.CoordList
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

        self.pointings = pointings.pix(self.step_degree)  

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
        oshape = (len(self.pointings), self.instr.n_slit, len(self.instr.wavel_axis), ceil(self.slicer.npix_slit_alpha_width / self.srf))

        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        self.local_cube_shape = (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr =python_utils.udft.ir2fr(np.ones((self.srf, 1)), self.local_cube_shape[1:])[np.newaxis, ...]
        self.wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=self.instr,
                wslice=slice(0, len(self.wavelength_axis), None)
                )

        decal = np.zeros(self.local_cube_shape[1:])
        dsi = int((self.srf-1)/2)
        dsj = 0 # int((self.n_pix_beta_slit -1) /2)
        decal[-dsi, -dsj] = np.sqrt(self.local_cube_shape[1]*self.local_cube_shape[2])
        self.decalf = jax_utils.dft(decal)
        """
        decal = np.zeros(self.local_cube_shape[1:])
        dsi = int((self.srf-1)/2)
        dsj = 0 # int((self.n_pix_beta_slit -1) /2)
        decal[-dsi, -dsj] = np.sqrt(self.local_cube_shape[1]*self.local_cube_shape[2])
        decalf = jax_utils.dft(decal)
        sum_cube = jax_utils.idft(
                jax_utils.dft(gridded) * self._otf_sr * decalf,
                self.local_cube_shape[1:],
            )
        res = sum_cube[:, ::srf, ::self.n_pix_beta_slit]

        Adjoint :
        local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
        local_cube[:,::self.srf, ::self.n_pix_beta_slit] = someArray
        sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*decalf.conj(), self.local_cube_shape[1:])

        """

    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step_degree)
    
    @property
    def alpha_step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]
    
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

    def gridding(self, blurred_cube: array, pointing: instru.Coord) -> array:
        local_alpha_coord, local_beta_coord = (self.instr.fov + pointing).local2global(
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
        
        return gridded
    
    def gridding_t(self, local_cube: array, pointing: instru.Coord) -> array:
        alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
                self.alpha_axis, self.beta_axis
                )
        
        optimized_global_coords = np.vstack(
                [
                    alpha_coord.ravel(),
                    beta_coord.ravel()
                ]
                ).T

        global_cube = cython_utils.interpn_local2cube(self.wavelength_axis, 
                                                self.local_alpha_axis.ravel(), 
                                                self.local_beta_axis.ravel(), 
                                                np.array(local_cube, dtype=np.float64), 
                                                optimized_global_coords, 
                                                self.cube_shape)
        return global_cube

    
    def forward(self, maps: np.ndarray) -> np.ndarray:
        # T
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))
        allsliced = np.zeros(self.oshape)
        
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred_cube, pointing) 
            sum_cube = jax_utils.idft(
                jax_utils.dft(gridded) * self._otf_sr*self.decalf,
                self.local_cube_shape[1:],
            )
            for slit_idx in range(self.instr.n_slit):
                #L
                sliced = self.slicer.slicing(sum_cube, slit_idx)
                # SigR
                blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.wpsf)
                allsliced[p_idx, slit_idx] = blurred_sliced_subsampled[:, ::self.srf]
        return allsliced
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        global_cube = np.zeros(self.cube_shape)
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            inarray[p_idx, slit_idx],
                            axis=2,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=2,
                    )
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
                blurred_t_sliced[:, ::self.srf, :] = jax_utils.wblur_t(oversampled_sliced, self.wpsf.conj())
                local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*self.decalf.conj(), self.local_cube_shape[1:])

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing)
            global_cube += degridded

        blurred_t_cube = jax_utils.idft(jax_utils.dft(global_cube) * self.sotf.conj(), (self.ishape[1], self.ishape[2]))
        maps = jax_utils.lmm_cube2maps(blurred_t_cube, self.templates).reshape(self.ishape)
        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)

    def make_mask(self, maps):
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        allsliced = np.zeros(self.oshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(cube, pointing) 
            sum_cube = jax_utils.idft(
                jax_utils.dft(gridded) * self._otf_sr,
                self.local_cube_shape[1:],
            )
            for slit_idx in range(self.instr.n_slit):
                #L
                sliced = self.slicer.slicing(sum_cube, slit_idx)
                # SigR
                blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.wpsf)
                allsliced[p_idx, slit_idx] = blurred_sliced_subsampled[:, : self.oshape[3] * self.srf : self.srf]


        global_cube = np.zeros(self.cube_shape)
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((len(self.wavelength_axis),  len(self.local_alpha_axis), len(self.local_beta_axis)))
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            allsliced[p_idx, slit_idx],
                            axis=2,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=2,
                    )
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
                blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced, self.wpsf.conj())
                local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis)))

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj(), self.local_cube_shape[1:])

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing)
            global_cube += degridded


        mask = np.ones_like(global_cube)
        zeros = np.where(global_cube == 0.)
        mask[zeros] = 0

        return mask