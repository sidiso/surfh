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
from surfh.DottestModels import MCMO_SigRLSCT_Channel_Model
from math import ceil


from typing import List, Tuple
from numpy import ndarray as array

import jax
from functools import partial

"""
Multi-Observation Multi Channel for the spectro model :

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
                instrs: List[instru.IFU],  
                step_degree: float,
                pointings: instru.CoordList
                ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.instrs = instrs
        self.step_degree = step_degree


        self.list_local_alpha_axis = []
        self.list_local_beta_axis = []
        self.list_slicer = []
        self.list_wslice = [instr.wslice(wavelength_axis, 0.1) for instr in self.instrs]
        for chan_idx, chan in enumerate(self.instrs):
            local_alpha_axis, local_beta_axis = chan.fov.local_coords(step_degree, 0, 0)#5* step_degree, 5* step_degree)
            self.list_local_alpha_axis.append(local_alpha_axis)
            self.list_local_beta_axis.append(local_beta_axis)  
            self.list_slicer.append(slicer.Slicer(chan, 
                                    wavelength_axis = self.wavelength_axis[self.list_wslice[chan_idx]], 
                                    alpha_axis = self.alpha_axis, 
                                    beta_axis = self.beta_axis, 
                                    local_alpha_axis = local_alpha_axis, 
                                    local_beta_axis = local_beta_axis))

        self.pointings = pointings.pix(self.step_degree)  
       
       
        # Super resolution factor (in alpha dim) for all channels
        self.srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step_degree*3600, # Conversion in arcsec
        )
        print(f'Super Resolution factor is set to {self.srfs}')

        self.list_wslice = [instr.wslice(wavelength_axis, 0.1) for instr in self.instrs]

        
        # Templates (4, Nx, Ny)
        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        self.instrs_oshape = [(len(pointings), instrs[idx].n_slit, 
                          len(instrs[idx].wavel_axis), 
                          ceil(self.list_slicer[idx].npix_slit_alpha_width / self.srfs[idx])) 
                          for idx in range(len(instrs))]
        
        self._idx = np.cumsum([0] + [np.prod(oshape) for oshape in self.instrs_oshape])
        oshape = (self._idx[-1],)
        
        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
        # self.local_cube_shape = (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))
        self.list_local_im_shape = [(len(self.list_local_alpha_axis[idx]), len(self.list_local_beta_axis[idx])) for idx, _ in enumerate(self.instrs)]
        self.imshape = (len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)

        # # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self.list_otf_sr = [python_utils.udft.ir2fr(
                                            np.ones((self.srfs[idx], 1)), self.list_local_im_shape[idx]
                                            )[np.newaxis, ...] 
                                            for idx, _ in enumerate(self.instrs)]
        self.list_wpsf = [self._wpsf(length=self.list_slicer[idx].npix_slit_beta_width,
                step=self.beta_step,
                wavel_axis=self.wavelength_axis,
                instr=instr,
                wslice=self.list_wslice[idx]
                ) for idx, instr in enumerate(self.instrs)]

    
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

    def gridding(self, blurred_cube: array, pointing: instru.Coord, chan_idx: int) -> array:

        local_alpha_coord, local_beta_coord = (self.instrs[chan_idx].fov + pointing).local2global(
                                                        self.list_local_alpha_axis[chan_idx], self.list_local_beta_axis[chan_idx]
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
                                                   (blurred_cube.shape[0], len(self.list_local_alpha_axis[chan_idx]), len(self.list_local_beta_axis[chan_idx])))
        
        return gridded
    
    def gridding_t(self, local_cube: array, pointing: instru.Coord, chan_idx: int) -> array:
        alpha_coord, beta_coord = (self.instrs[chan_idx].fov + pointing).global2local(
                self.alpha_axis, self.beta_axis
                )
        
        optimized_global_coords = np.vstack(
                [
                    alpha_coord.ravel(),
                    beta_coord.ravel()
                ]
                ).T

        global_cube = cython_utils.interpn_local2cube(np.arange(local_cube.shape[0]), 
                                                self.list_local_alpha_axis[chan_idx].ravel(), 
                                                self.list_local_beta_axis[chan_idx].ravel(), 
                                                np.array(local_cube, dtype=np.float64), 
                                                optimized_global_coords, 
                                                (len(np.arange(local_cube.shape[0])), len(self.alpha_axis), len(self.beta_axis)))
        return global_cube

    def forward(self, maps: np.ndarray) -> np.ndarray:
        # T
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))
        out = np.zeros(self.oshape)
        for ch_idx, chan in enumerate(self.instrs):
            chan_out = np.zeros(self.instrs_oshape[ch_idx])
            for p_idx, pointing in enumerate(self.pointings):
                gridded = self.gridding(blurred_cube[self.list_wslice[ch_idx]], pointing, ch_idx) 
                sum_cube = jax_utils.idft(
                    jax_utils.dft(gridded) * self.list_otf_sr[ch_idx],
                    self.list_local_im_shape[ch_idx],
                )
                for slit_idx in range(chan.n_slit):
                    #L
                    sliced = self.list_slicer[ch_idx].slicing(sum_cube, slit_idx)
                    # SigR
                    blurred_sliced_subsampled = jax_utils.wblur_subSampling(sliced, self.list_wpsf[ch_idx])
                    chan_out[p_idx, slit_idx] = blurred_sliced_subsampled[:, : self.instrs_oshape[ch_idx][3] * self.srfs[ch_idx] : self.srfs[ch_idx]]
            out[self._idx[ch_idx] : self._idx[ch_idx + 1]] = chan_out.ravel()
        return out
   
    
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        global_cube = np.zeros(self.cube_shape)

        for ch_idx, chan in enumerate(self.instrs):
            inter_cube = np.zeros((self.list_wslice[ch_idx].stop-self.list_wslice[ch_idx].start, len(self.alpha_axis), len(self.beta_axis)))
            for p_idx, pointing in enumerate(self.pointings):
                local_cube = np.zeros((self.list_wslice[ch_idx].stop-self.list_wslice[ch_idx].start,  
                                       len(self.list_local_alpha_axis[ch_idx]), 
                                       len(self.list_local_beta_axis[ch_idx])))
                for slit_idx in range(chan.n_slit):
                    oversampled_sliced = np.repeat(
                            np.expand_dims(
                                np.reshape(inarray[self._idx[ch_idx] : self._idx[ch_idx + 1]], self.instrs_oshape[ch_idx])[p_idx, slit_idx],
                                axis=2,
                            ),
                            self.list_slicer[ch_idx].npix_slit_beta_width,
                            axis=2,
                        )
                    blurred_t_sliced = np.zeros(self.list_slicer[ch_idx].get_slit_shape_t())
                    blurred_t_sliced[:,: self.instrs_oshape[ch_idx][3] * self.srfs[ch_idx] : self.srfs[ch_idx],:] = jax_utils.wblur_t(oversampled_sliced, self.list_wpsf[ch_idx].conj())
                    tmp = self.list_slicer[ch_idx].slicing_t(blurred_t_sliced, slit_idx, 
                                                                     (self.list_wslice[ch_idx].stop-self.list_wslice[ch_idx].start, 
                                                                      len(self.list_local_alpha_axis[ch_idx]), 
                                                                      len(self.list_local_beta_axis[ch_idx])))

                    local_cube += tmp

                sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self.list_otf_sr[ch_idx].conj(), 
                                            self.list_local_im_shape[ch_idx])

                degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), pointing, ch_idx)
                inter_cube += degridded

            global_cube[self.list_wslice[ch_idx]] += inter_cube
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
    

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
"""
Multi-Observation Multi Channel for the spectro model :

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
class spectroSigRLSCT_NN(LinOp):
    def __init__(
                self,
                sotf: array,
                templates: array,
                alpha_axis: array,
                beta_axis: array,
                wavelength_axis: array,
                instrs: List[instru.IFU],  
                step_degree: float,
                pointings: instru.CoordList
                ):
        self.sotf = sotf
        self.templates = templates
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.step_degree = step_degree
        self.instrs = [instr.pix(self.step_degree) for instr in instrs]


        self.list_local_alpha_axis = []
        self.list_local_beta_axis = []
        self.list_slicer = []
        self.list_wslice = [instr.wslice(wavelength_axis, 0.1) for instr in self.instrs]
        for chan_idx, chan in enumerate(self.instrs):
            local_alpha_axis, local_beta_axis = chan.fov.local_coords(step_degree, 5* step_degree, 5* step_degree)
            self.list_local_alpha_axis.append(local_alpha_axis)
            self.list_local_beta_axis.append(local_beta_axis)  
            self.list_slicer.append(slicer.Slicer(chan, 
                                    wavelength_axis = self.wavelength_axis[self.list_wslice[chan_idx]], 
                                    alpha_axis = self.alpha_axis, 
                                    beta_axis = self.beta_axis, 
                                    local_alpha_axis = local_alpha_axis, 
                                    local_beta_axis = local_beta_axis))
            
        self.pointings = pointings.pix(self.step_degree)  
       
        print(self.step_degree*3600)
        print(chan.det_pix_size)
        # Super resolution factor (in alpha dim) for all channels
        self.srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step_degree*3600, # Conversion in arcsec
        )
        print(f'Super Resolution factor is set to {self.srfs}')

        self.list_wslice = [instr.wslice(wavelength_axis, 0.1) for instr in self.instrs]

        
        # Templates (4, Nx, Ny)
        ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        
        # 4D array [Nslit, L, alpha_slit, beta_slit]
        self.instrs_oshape = [(len(pointings), instrs[idx].n_slit, 
                          len(instrs[idx].wavel_axis), 
                          ceil(self.list_slicer[idx].npix_slit_alpha_width / self.srfs[idx])) 
                          for idx in range(len(instrs))]
        
        self._idx = np.cumsum([0] + [np.prod(oshape) for oshape in self.instrs_oshape])
        oshape = (self._idx[-1],)

        self.list_instr_cube_shape = [(self.list_wslice[idx].stop - self.list_wslice[idx].start, 
                                        len(self.alpha_axis), 
                                        len(self.beta_axis)) for idx, _ in enumerate(self.instrs) for p in range(len(self.pointings))]
        
        self.cube_shape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))

        # self.local_cube_shape = (len(self.wavelength_axis), len(self.local_alpha_axis), len(self.local_beta_axis))
        self.list_local_im_shape = [(len(self.list_local_alpha_axis[idx]), len(self.list_local_beta_axis[idx])) for idx, _ in enumerate(self.instrs)]
        self.imshape = (len(alpha_axis), len(beta_axis))

        super().__init__(ishape=ishape, oshape=oshape)
        # self.ishape = tuple(ishape)
        # self.oshape = tuple(oshape)

        # # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self.list_otf_sr = [python_utils.udft.ir2fr(
                                            np.ones((self.srfs[idx], 1)), self.list_local_im_shape[idx]
                                            )[np.newaxis, ...] 
                                            for idx, _ in enumerate(self.instrs)]
        
        # self.nmask = np.zeros((len(self.instrs), len(self.pointings), self.imshape[0], self.imshape[1]))
        # self.precompute_mask()

        # self.list_gridding_indexes = []
        # self.precompute_griding_indexes()

        # self.list_gridding_t_indexes = []
        # self.precompute_griding_t_indexes()

        self.channels = [
            MCMO_SigRLSCT_Channel_Model.Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavelength_axis,
                srf,
                pointings,
                step_degree
            )
            for srf, instr in zip(self.srfs, instrs)
        ]


    
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

    def precompute_griding_indexes(self):
        for ch_idx, chan in enumerate(self.instrs):
            for p_idx, pointing in enumerate(self.pointings):
                dummy_cube = np.ones(self.list_instr_cube_shape[ch_idx*len(self.pointings) + p_idx])
                local_alpha_coord, local_beta_coord = (self.instrs[ch_idx].fov + pointing).local2global(
                                                                self.list_local_alpha_axis[ch_idx], self.list_local_beta_axis[ch_idx]
                                                                )

                test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
                test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis)) 
                # S
                wavel_idx = np.arange(self.list_wslice[ch_idx].stop - self.list_wslice[ch_idx].start)

                indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), 
                                                                dummy_cube[0].ravel(), 
                                                                (local_alpha_coord, local_beta_coord))
                
                wavel_indexes = np.tile(indexes, 
                                        (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*dummy_cube[0].size )
 
                self.list_gridding_indexes.append(wavel_indexes)

    
    def precompute_griding_t_indexes(self):
        for ch_idx, chan in enumerate(self.instrs):
            for p_idx, pointing in enumerate(self.pointings):
                dummy_cube = np.ones((self.list_wslice[ch_idx].stop-self.list_wslice[ch_idx].start,  
                                       len(self.list_local_alpha_axis[ch_idx]), 
                                       len(self.list_local_beta_axis[ch_idx])))
                

                local_alpha_coord, local_beta_coord = (self.instrs[ch_idx].fov + pointing).local2global(
                                                                self.list_local_alpha_axis[ch_idx], self.list_local_beta_axis[ch_idx]
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

        
    


    def NN_gridding(self, blurred_cube: array, pointing: instru.Coord, chan_idx: int, p_idx: int) -> array:
        local_alpha_coord, local_beta_coord = (self.instrs[chan_idx].fov + pointing).local2global(
                                                        self.list_local_alpha_axis[chan_idx], self.list_local_beta_axis[chan_idx]
                                                        )
        # S
        wavel_idx = np.arange(self.list_wslice[chan_idx].stop - self.list_wslice[chan_idx].start)
        wavel_indexes = self.list_gridding_indexes[chan_idx*len(self.pointings) + p_idx]
        gridded = blurred_cube.ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])

        return gridded
    


    def NN_gridding_t(self, local_cube: array, pointing: instru.Coord, chan_idx: int, p_idx: int) -> array:

        wavel_indexes_t = self.list_gridding_t_indexes[chan_idx*len(self.pointings) + p_idx]
        degridded = local_cube.ravel()[wavel_indexes_t].reshape(local_cube.shape[0], len(self.alpha_axis), len(self.beta_axis))

        return degridded

    def forward(self, maps):
        # T
        cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        # C
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))
        out = np.zeros(self.oshape)
        for ch_idx, chan in enumerate(self.channels):
            out[self._idx[ch_idx] : self._idx[ch_idx + 1]] = chan.forward(blurred_cube)
        return out
   

    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        global_cube = np.zeros(self.cube_shape)
        print(global_cube.shape)

        for ch_idx, chan in enumerate(self.channels):
            global_cube[self.list_wslice[ch_idx]] += chan.adjoint(inarray[self._idx[ch_idx] : self._idx[ch_idx + 1]],)

        blurred_t_cube = jax_utils.idft(jax_utils.dft_mult(global_cube, self.sotf.conj()), (self.ishape[1], self.ishape[2]))
        maps = jax_utils.lmm_cube2maps(blurred_t_cube, self.templates).reshape(self.ishape)
        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)


    def precompute_mask(self):
        rin = np.ones(self.ishape)
        cube_rin = np.array(jax_utils.lmm_maps2cube(rin, self.templates)).astype(np.float64)
        for chan_idx, chan in enumerate(self.instrs):
            chan_out = np.zeros(self.instrs_oshape[chan_idx])
            for p_idx, pointing in enumerate(self.pointings):

                local_alpha_coord, local_beta_coord = (self.instrs[chan_idx].fov + pointing).local2global(
                                                                self.list_local_alpha_axis[chan_idx], self.list_local_beta_axis[chan_idx]
                                                                )

                test_cube_alpha_axis = np.tile(self.alpha_axis, len(self.alpha_axis))
                test_cube_beta_axis= np.repeat(self.beta_axis, len(self.beta_axis)) 
                # S
                wavel_idx = np.arange(self.list_wslice[chan_idx].stop - self.list_wslice[chan_idx].start)
                indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), 
                                                                cube_rin[0].ravel(), 
                                                                (local_alpha_coord, local_beta_coord))
                wavel_indexes = np.tile(indexes, 
                                        (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*cube_rin[0].size )
                gridded = cube_rin[self.list_wslice[chan_idx]].ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])


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
                
                                
                self.nmask[chan_idx, p_idx] = nmask


    def make_small_mask(self):
        ones_mask = np.ones(self.ishape)
        cube = jax_utils.lmm_maps2cube(ones_mask, self.templates).reshape(self.cube_shape)
        # C
        out = np.zeros(self.oshape)
        out[self._idx[0] : self._idx[0 + 1]] = self.channels[0].forward(cube)

        global_cube = np.zeros(self.cube_shape)

        global_cube[self.list_wslice[0]] += self.channels[0].adjoint(out[self._idx[0] : self._idx[0 + 1]],)

        maps = jax_utils.lmm_cube2maps(global_cube, self.templates).reshape(self.ishape)
        return maps


        