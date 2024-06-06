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

import jax
from functools import partial


class Channel():
    def __init__(
        self,
        instr: instru.IFU,
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        srf: int,
        pointings: instru.CoordList,
        step_degree: float
    ):
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.step_degree = step_degree
        self.global_wavelength_axis = wavel_axis
        self.srf = srf

        self.instr = instr.pix(self.step_degree)
        self.pointings = pointings.pix(self.step_degree)

        local_alpha_axis, local_beta_axis = self.instr.fov.local_coords(step_degree, 0, 0)
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        

        self.slicer = slicer.Slicer(self.instr, 
                                    wavelength_axis = self.global_wavelength_axis[self.wslice], 
                                    alpha_axis = self.alpha_axis, 
                                    beta_axis = self.beta_axis, 
                                    local_alpha_axis = local_alpha_axis, 
                                    local_beta_axis = local_beta_axis)
        
        self.ishape = (len(self.global_wavelength_axis), len(alpha_axis), len(beta_axis))

        self.oshape = (len(pointings), self.instr.n_slit, 
                          len(self.instr.wavel_axis), 
                          ceil(self.slicer.npix_slit_alpha_width / self.srf)) 

        # 4D array [NPointing, Nslit, L, alpha_slit]
        instrs_oshape = (len(pointings), self.instr.n_slit, 
                          len(self.instr.wavel_axis), 
                          ceil(self.slicer.npix_slit_alpha_width / self.srf)
                        )

        self.local_im_shape = (len(self.local_alpha_axis), len(self.local_beta_axis))
        self.imshape = (len(alpha_axis), len(beta_axis))

        self.instr_cube_shape = (self.wslice.stop-self.wslice.start, 
                                        len(self.alpha_axis), 
                                        len(self.beta_axis))


        self._otf_sr = python_utils.udft.ir2fr(
                                            np.ones((self.srf, 1)), self.local_im_shape
                                            )[np.newaxis, ...] 
                                           
        self.wpsf = self._wpsf(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.global_wavelength_axis,
                instr=self.instr,
                wslice=self.wslice
                )

        self.nmask = np.zeros((len(self.pointings), self.imshape[0], self.imshape[1]))
        self.precompute_mask()

        self.list_gridding_indexes = []
        self.precompute_griding_indexes()

        self.list_gridding_t_indexes = []
        self.precompute_griding_t_indexes()



    @property
    def wslice(self) -> slice:
        """The wavelength slice of input that match instr with 0.1 μm of margin."""
        wavel_axis = self.global_wavelength_axis
        return self.instr.wslice(wavel_axis, 0.1)
    
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

    def NN_gridding(self, blurred_cube: array, p_idx: int) -> array:
        wavel_indexes = self.list_gridding_indexes[p_idx]
        gridded = blurred_cube.ravel()[wavel_indexes].reshape(self.instr_cube_shape[0], 
                                                              len(self.local_alpha_axis), 
                                                              len(self.local_beta_axis))
        return gridded

    def NN_gridding_t(self, local_cube: array, p_idx: int) -> array:
        wavel_indexes_t = self.list_gridding_t_indexes[p_idx]
        degridded = local_cube.ravel()[wavel_indexes_t].reshape(local_cube.shape[0],
                                                                len(self.alpha_axis), 
                                                                len(self.beta_axis))
        return degridded

    @jax.jit
    def forward(self, blurred_cube):
        chan_out = np.zeros(self.oshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.NN_gridding(blurred_cube[self.wslice], p_idx) 
            sum_cube = jax_utils.idft(
                jax_utils.dft_mult(gridded, self._otf_sr),
                self.local_im_shape,
            )
            for slit_idx in range(self.instr.n_slit):
                #L
                sliced = self.slicer.slicing(sum_cube, slit_idx)
                # SigR
                blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.wpsf)
                chan_out[p_idx, slit_idx] = blurred_sliced_subsampled[:, : self.oshape[3] * self.srf : self.srf]

        return chan_out.ravel()

    def adjoint(self, inarray: np.ndarray) -> np.ndarray:

        inter_cube = np.zeros((self.wslice.stop-self.wslice.start, len(self.alpha_axis), len(self.beta_axis)))
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((self.wslice.stop-self.wslice.start,
                                   self.local_im_shape[0],
                                   self.local_im_shape[1]))
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            np.reshape(inarray, 
                                       self.oshape)[p_idx, slit_idx],
                            axis=2,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=2,
                    )
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
                blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced, self.wpsf.conj())
                tmp = self.slicer.slicing_t(blurred_t_sliced, slit_idx, (self.wslice.stop-self.wslice.start,
                                                                        self.local_im_shape[0],
                                                                        self.local_im_shape[1]))
                local_cube += tmp

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj(), 
                                        self.local_im_shape)

            degridded = self.NN_gridding_t(np.array(sum_t_cube, dtype=np.float64), p_idx)
            inter_cube += degridded*self.nmask[p_idx]

        return inter_cube



    def precompute_mask(self):
        cube_rin = np.ones((len(self.global_wavelength_axis), len(self.alpha_axis), len(self.beta_axis)))
        for p_idx, pointing in enumerate(self.pointings):

            local_alpha_coord, local_beta_coord = (self.instr.fov + pointing).local2global(
                                                            self.local_alpha_axis, self.local_beta_axis
                                                            )

            test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
            test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis)) 
            # S
            wavel_idx = np.arange(self.wslice.stop - self.wslice.start)
            indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), 
                                                            cube_rin[0].ravel(), 
                                                            (local_alpha_coord, local_beta_coord))
            wavel_indexes = np.tile(indexes, 
                                    (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*cube_rin[0].size )
            gridded = cube_rin[self.wslice].ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])


            indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()),
                                                                gridded[0].ravel(), 
                                                                (test_cube_alpha_axis.reshape(self.imshape[0],self.imshape[1]), test_cube_beta_axis.reshape(self.imshape[0], self.imshape[1])))
            wavel_indexes_t = np.tile(indexes_t, 
                                    (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes_t)) + ((wavel_idx[...,np.newaxis])*local_alpha_coord.shape[0]* local_alpha_coord.shape[1] )
            
            degridded = gridded.ravel()[wavel_indexes_t].reshape(gridded.shape[0], len(self.alpha_axis), len(self.beta_axis))


            mask = np.zeros_like(degridded[0])
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
            
                            
            self.nmask[p_idx] = nmask

    def precompute_griding_indexes(self):
        for p_idx, pointing in enumerate(self.pointings):
            dummy_cube = np.ones(self.instr_cube_shape)
            local_alpha_coord, local_beta_coord = (self.instr.fov + pointing).local2global(
                                                            self.local_alpha_axis, self.local_beta_axis
                                                            )

            test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
            test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis)) 
            # S
            wavel_idx = np.arange(self.wslice.stop - self.wslice.start)

            indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), 
                                                            dummy_cube[0].ravel(), 
                                                            (local_alpha_coord, local_beta_coord))
            
            wavel_indexes = np.tile(indexes, 
                                    (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*dummy_cube[0].size )

            self.list_gridding_indexes.append(wavel_indexes)

    def precompute_griding_t_indexes(self):
        for p_idx, pointing in enumerate(self.pointings):
            dummy_cube = np.ones((self.wslice.stop-self.wslice.start,  
                                    len(self.local_alpha_axis), 
                                    len(self.local_beta_axis)))
            

            local_alpha_coord, local_beta_coord = (self.instr.fov + pointing).local2global(
                                                            self.local_alpha_axis, self.local_beta_axis
                                                            )

            test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
            test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis))  

            wavel_idx = np.arange(dummy_cube.shape[0])

            indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()),
                                                                dummy_cube[0].ravel(), 
                                                                (test_cube_alpha_axis.reshape(self.imshape[0],self.imshape[1]), test_cube_beta_axis.reshape(self.imshape[0], self.imshape[1])))
            
            
            wavel_indexes_t = np.tile(indexes_t, 
                                    (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes_t)) + ((wavel_idx[...,np.newaxis])*local_alpha_coord.shape[0]* local_alpha_coord.shape[1] )
            self.list_gridding_t_indexes.append(wavel_indexes_t)
