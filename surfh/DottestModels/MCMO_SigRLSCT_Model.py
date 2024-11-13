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
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.wavelength_axis = wavelength_axis
        self.step_degree = step_degree
        self.instrs = [instr.pix(self.step_degree) for instr in instrs]

        self.templates = templates
        
        if self.templates is None:
            self.lmm = False
        else:
            self.lmm = True

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
            
        self.pointings = pointings
       
        # Super resolution factor (in alpha dim) for all channels
        self.srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step_degree*3600, # Conversion in arcsec
        )

        # Templates (4, Nx, Ny)
        if self.lmm:
            ishape = (self.templates.shape[0], len(alpha_axis), len(beta_axis))
        else:
            ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))

        # 4D array [Nslit, L, alpha_slit, beta_slit]
        self.instrs_oshape = [(len(pointings[0]), instrs[idx].n_slit, 
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

        # # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self.list_otf_sr = [python_utils.udft.ir2fr(
                                            np.ones((self.srfs[idx], 1)), self.list_local_im_shape[idx]
                                            )[np.newaxis, ...] 
                                            for idx, _ in enumerate(self.instrs)]
        
        self.channels = [
            MCMO_SigRLSCT_Channel_Model.Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavelength_axis,
                srf,
                pointings[it],
                step_degree
            )
            for it, (srf, instr) in enumerate(zip(self.srfs, instrs))
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

    def forward(self, maps):
        # T
        if self.lmm:
            cube = jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.cube_shape)
        else:
            cube = maps

        # C
        # blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.ishape[1], self.ishape[2]))
        out = np.zeros(self.oshape)
        for ch_idx, chan in enumerate(self.channels):
            out[self._idx[ch_idx] : self._idx[ch_idx + 1]] = chan.forward(cube)
        return out
   

    def adjoint(self, inarray: np.ndarray) -> np.ndarray:
        global_cube = np.zeros(self.cube_shape)
        for ch_idx, chan in enumerate(self.channels):
            global_cube[self.list_wslice[ch_idx]] += chan.adjoint(inarray[self._idx[ch_idx] : self._idx[ch_idx + 1]],)

        # blurred_t_cube = jax_utils.idft(jax_utils.dft_mult(global_cube, self.sotf.conj()), (self.ishape[1], self.ishape[2]))

        if self.lmm:
            maps = jax_utils.lmm_cube2maps(global_cube, self.templates).reshape(self.ishape)
        else:
            maps = global_cube

        return maps
    
    def cubeTomaps(self, cube):
        return jax_utils.lmm_cube2maps(cube, self.templates)
    
    def mapsToCube(self, maps):
        return jax_utils.lmm_maps2cube(maps, self.templates)


    def project_FOV(self):
        plt.figure()
        for ch_idx, chan in enumerate(self.channels):
            chan.project_FOV()
            alpha = [np.min(self.alpha_axis), np.max(self.alpha_axis), np.min(self.alpha_axis),np.max(self.alpha_axis)]
            beta = [np.min(self.beta_axis), np.min(self.beta_axis), np.max(self.beta_axis),np.max(self.beta_axis)]
            print(alpha, beta)
            plt.plot(alpha, beta, 'o', label='Reference')

            plt.legend()
        plt.show()
        return

    def test_vizual_projection(self, data):

        global_cube = np.zeros(self.cube_shape)
        cube_list = list()
        wave_list = list()
        for ch_idx, chan in enumerate(self.channels):
            cube_list.append(chan.test_vizual_projection(data[self._idx[ch_idx] : self._idx[ch_idx + 1]],))
            wave_list.append(chan.instr.wavel_axis)
        return cube_list, wave_list


    def real_data_janskySR_to_jansky(self, data : np.array):
        """
        Normalize raw 2D real data flux from Jy/sr to Jy
        """
        normalized_data = np.zeros_like(data)
        for ch_idx, chan in enumerate(self.channels):
            chan_data = data[self._idx[ch_idx] : self._idx[ch_idx + 1]]
            chan_data = chan_data.reshape(self.instrs_oshape[ch_idx])
            for slit in range(self.instrs_oshape[ch_idx][1]):
                slices = chan.slicer.get_slit_slices(slit_idx=slit)
                weights = chan.slicer.get_slit_weights(slit_idx=slit, slices=slices)  
                chan_data[:,slit,:,:] = chan_data[:,slit,:,:]*(np.sum(weights[0,0,:]))*self.srfs[ch_idx]
            normalized_data[self._idx[ch_idx] : self._idx[ch_idx + 1]] = chan_data.ravel()

        return normalized_data
        


