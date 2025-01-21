import pytest
import numpy as np
from aljabr import LinOp, dottest
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.Models import slicer_new as slicer
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils, nearest_neighbor_interpolation
from astropy import units as u
from astropy.coordinates import Angle
from numpy.random import standard_normal as randn 

from surfh.Models import instru
from math import ceil
import operator as op

from typing import List, Tuple
from numpy import ndarray as array

import jax
from jax import numpy as jnp
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

        local_alpha_axis, local_beta_axis = self.instr.fov.local_coords(step_degree, 
                                                                        alpha_margin=5* step_degree, 
                                                                        beta_margin=5* step_degree)
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        

        self.slicer = slicer.Slicer(self.instr, 
                                    wavelength_axis = self.global_wavelength_axis, #[self.wslice], 
                                    alpha_axis = self.alpha_axis, 
                                    beta_axis = self.beta_axis, 
                                    local_alpha_axis = local_alpha_axis, 
                                    local_beta_axis = local_beta_axis,
                                    srf = self.srf)
        
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
        
        self.wpsf_dirac = self._wpsf_dirac(length=self.slicer.npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.global_wavelength_axis,
                instr=self.instr,
                wslice=self.wslice
                )

        self.local_cube_shape = (len(self.global_wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))

        self.slices_shape = (len(pointings), instr.n_slit, ceil(self.slicer.npix_slit_alpha_width / self.srf))


        decal = np.zeros(self.local_cube_shape[1:])
        dsi = int((self.srf-1)/2)
        dsj = 0 # int((self.n_pix_beta_slit -1) /2)
        decal[-dsi, -dsj] = np.sqrt(self.local_cube_shape[1]*self.local_cube_shape[2])
        self.decalf = jax_utils.dft(decal)
        

        # self.nmask = np.zeros((len(self.pointings), self.imshape[0], self.imshape[1]))
        # self.precompute_mask()

        # self.list_gridding_indexes = []
        # self.precompute_griding_indexes()

        # self.list_gridding_t_indexes = []
        # self.precompute_griding_t_indexes()



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

    def _wpsf_dirac(self, length: int, step: float, wavel_axis: np.ndarray, instr: instru.IFU, wslice) -> array:
        """Return spectral PSF"""
        # ∈ [0, β_s]
        beta_in_slit = np.arange(0, length) * step
        wpsf = instr.spectral_psf(
                        beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
                        wavel_axis[wslice],
                        arcsec2micron=instr.wavel_step / instr.det_pix_size,
                        type='dirac',
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
        gridded = cython_utils.interpn_cube2local(np.arange(blurred_cube.shape[0]).astype(np.float64), 
                                                   self.alpha_axis, 
                                                   self.beta_axis, 
                                                   np.array(blurred_cube).astype(np.float64), 
                                                   optimized_local_coords, 
                                                   (blurred_cube.shape[0], len(self.local_alpha_axis), len(self.local_beta_axis)))
        
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

        global_cube = cython_utils.interpn_local2cube(np.arange(local_cube.shape[0]), 
                                                self.local_alpha_axis.ravel(), 
                                                self.local_beta_axis.ravel(), 
                                                np.array(local_cube, dtype=np.float64), 
                                                optimized_global_coords, 
                                                (len(np.arange(local_cube.shape[0])), len(self.alpha_axis), len(self.beta_axis)))
        return global_cube

    def NN_gridding(self, blurred_cube: array, wavel_indexes: array) -> array:
        gridded = blurred_cube.ravel()[wavel_indexes].reshape(self.instr_cube_shape[0], 
                                                              len(self.local_alpha_axis), 
                                                              len(self.local_beta_axis))
        return gridded

    
    def NN_gridding_t(self, local_cube: array, wavel_indexes_t: array) -> array:
        degridded = local_cube.ravel()[wavel_indexes_t].reshape(local_cube.shape[0],
                                                                len(self.alpha_axis), 
                                                                len(self.beta_axis))
        return degridded

    
    def forward(self, blurred_cube):
        chan_out = np.zeros(self.oshape)
        for p_idx, pointing in enumerate(self.pointings):
            # gridded = self.NN_gridding(blurred_cube[self.wslice], self.list_gridding_indexes[p_idx]) 
            gridded = self.gridding(blurred_cube[self.wslice], pointing) 
            sum_cube = jax_utils.idft(
                jax_utils.dft_mult(gridded, self._otf_sr*self.decalf),
                self.local_im_shape,
            )
            for slit_idx in range(self.instr.n_slit):
                #L
                sliced = self.slicer.slicing(sum_cube, slit_idx)
                # SigR
                # blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.wpsf)[:, : self.oshape[3] * self.srf : self.srf]
                chan_out[p_idx, slit_idx] = jax_utils.wblur_subSampling(sliced, self.wpsf)[:, : self.oshape[3] * self.srf : self.srf]

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

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing)
            inter_cube += degridded

        return inter_cube

    def sliceToCube(self, data):
        inter_cube = np.zeros((len(self.global_wavelength_axis), len(self.alpha_axis), len(self.beta_axis)))
        local_cube = np.zeros((self.wslice.stop-self.wslice.start,
                                self.local_im_shape[0],
                                self.local_im_shape[1]))
        for slit_idx in range(self.instr.n_slit):
            oversampled_sliced = np.repeat(
                    np.expand_dims(
                        np.reshape(data, 
                                    self.oshape)[0, slit_idx],
                        axis=2,
                    ),
                    self.slicer.npix_slit_beta_width,
                    axis=2,
                )
            blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())

            blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced.astype(np.float32), self.wpsf_dirac[:,:,:].conj())
            tmp = self.slicer.slicing_t(blurred_t_sliced, slit_idx, (self.wslice.stop-self.wslice.start,
                                                                    self.local_im_shape[0],
                                                                    self.local_im_shape[1]))
            local_cube += tmp

        sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*self.decalf.conj(), 
                                    self.local_im_shape)
        # plt.figure()
        # plt.imshow(sum_t_cube[150])
        # plt.show()


        sum_t_cube = np.array(sum_t_cube, dtype=np.float64)
        #sum_t_cube[:,0,:] = 0

        degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), self.pointings[0])
        inter_cube[self.wslice, ...] += degridded
        return inter_cube

    def realData_cubeToSlice(self, cube):
        slices = np.zeros(self.oshape[1:]) # Remove pointing dimension
        gridded = self.gridding(cube, instru.Coord(0, 0))
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicer.slicing(gridded, slit_idx)[:, : self.oshape[3] * self.srf : self.srf,:]
            slices[slit_idx, :, :] = sliced.sum(axis=2) # Only sum on the Beta axis
        return slices   

    def realData_sliceToCube(self, slices, cube_dim):
        blurred = np.zeros(cube_dim)
        gridded = np.zeros((cube_dim[0] , self.local_im_shape[0], self.local_im_shape[1]))
        for slit_idx in range(self.instr.n_slit):
            slices_slice = self.slicer.get_slit_slices(slit_idx)
            slice_alpha, slice_beta = slices_slice
            sliced = np.zeros((cube_dim[0], slice_alpha.stop - slice_alpha.start, slice_beta.stop - slice_beta.start))
            tmp = np.repeat(
                            np.expand_dims(
                                slices[slit_idx],
                                axis=2,
                                ),
                                self.slicer.npix_slit_beta_width,
                                axis=2,
                            )/self.slicer.npix_slit_beta_width
            tmp2 = tmp
            sliced[:, : cube_dim[0] * self.srf : self.srf] = tmp2
            tmp3 = self.slicer.slicing_t(sliced, slit_idx, (cube_dim[0],
                                                                        self.local_im_shape[0],
                                                                        self.local_im_shape[1]))
            gridded += tmp3

        sum_t_cube = jax_utils.idft(jax_utils.dft(np.array(gridded, dtype=np.float64)) * self._otf_sr.conj(), 
                                        self.local_im_shape)  
        blurred += self.gridding_t(sum_t_cube, instru.Coord(0, 0))
        return blurred



    def project_FOV(self):
        for p_idx, pointing in enumerate(self.pointings):
            f = self.instr.fov + pointing
            plt.plot(
                list(map(op.attrgetter("alpha"), f.vertices)) + [f.vertices[0].alpha],
                list(map(op.attrgetter("beta"), f.vertices)) + [f.vertices[0].beta],
                "-x",
                label=p_idx
            )


    def test_vizual_projection(self, data):
        inter_cube = np.zeros((len(self.global_wavelength_axis), len(self.alpha_axis), len(self.beta_axis)))
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((self.wslice.stop-self.wslice.start,
                                    self.local_im_shape[0],
                                    self.local_im_shape[1]))
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            np.reshape(data, 
                                        self.oshape)[p_idx, slit_idx],
                            axis=2,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=2,
                    )
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())

                blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced.astype(np.float32), self.wpsf_dirac[:,:,:].conj())
                tmp = self.slicer.slicing_t(blurred_t_sliced, slit_idx, (self.wslice.stop-self.wslice.start,
                                                                        self.local_im_shape[0],
                                                                        self.local_im_shape[1]))
                local_cube += tmp

            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)
            # plt.figure()
            # plt.imshow(sum_t_cube[150])
            # plt.show()


            sum_t_cube = np.array(sum_t_cube, dtype=np.float64)
            #sum_t_cube[:,0,:] = 0

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing)
            inter_cube[self.wslice, ...] += degridded
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

