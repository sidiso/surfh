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
from math import ceil, floor

import operator as op

from typing import List, Tuple
from numpy import ndarray as array

import jax
from functools import partial


class MRSBlurred(LinOp):
    def __init__(self,
                 sotf: array,
                 alpha_axis: array,
                 beta_axis: array,
                 instr: instru.IFU,
                 step_degree: float,
                 pointings: instru.CoordList):
        

        self.sotf = sotf
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.step_degree = step_degree
        self.instr = instr
        self.pointings = pointings

        self.srf = instru.get_srf(
            [instr.det_pix_size],
            self.step_degree*3600, # Conversion in arcsec
        )[0]


        local_alpha_axis, local_beta_axis = instr.fov.local_coords(step_degree, 5* step_degree, 5* step_degree)
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        self.local_im_shape = (len(self.local_alpha_axis), len(self.local_beta_axis))

        print("self.local_im_shape = ", self.local_im_shape)

        # Convolution kernel used to cumulate or dupplicate oversampled pixels during slitting operator L
        self._otf_sr = python_utils.udft.ir2fr(
                                            np.ones((self.srf, 1)), self.local_im_shape
                                            )

        ishape = (len(alpha_axis), len(beta_axis))
        self.slices_shape = (len(pointings), instr.n_slit, ceil(self.npix_slit_alpha_width / self.srf))
        oshape = [np.prod(self.slices_shape)]

        super().__init__(ishape, oshape)

        self.imshape = self.ishape
        decal = np.zeros(self.local_im_shape)
        dsi = int((self.srf-1)/2)
        dsj = 0 # int((self.n_pix_beta_slit -1) /2)
        decal[-dsi, -dsj] = np.sqrt(self.local_im_shape[0]*self.local_im_shape[1])
        self.decalf = jax_utils.dft(decal)

        
                                            

    @property
    def alpha_step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]
    
    @property   
    def beta_step(self) -> float:
        return self.beta_axis[1] - self.beta_axis[0]

    @property
    def slit_alpha_width(self):
        return self.instr.fov.alpha_width

    @property
    def npix_slit_alpha_width(self):
        """
        Number of oversampled pixel in alpha dim
        """
        step = self.local_alpha_axis[1] - self.local_alpha_axis[0]
        return int(ceil(self.slit_alpha_width / 2 / step)) - int(
            floor(-self.slit_alpha_width / 2 / step)
        )

    @property
    def slit_beta_width(self):
        """width of beta axis in slit local referential"""
        return self.instr.fov.beta_width/self.instr.n_slit

    @property
    def npix_slit_beta_width(self):
        """number of pixel for beta axis in slit local referential"""
        return int(ceil(self.slit_beta_width / (self.beta_axis[1] - self.beta_axis[0])))

    def slit_local_fov(self, slit_idx: int):
        """The FOV of slit `slit_idx` in local ref"""
        slit_fov = self.instr.slit_fov[slit_idx]
        return slit_fov.local + self.instr.slit_shift[slit_idx]

    def get_slit_shape_t(self):
        slices = self.get_slit_slices(0)
        return (
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
        )

    def get_slit_slices(self, slit_idx: int) -> Tuple[slice, slice]:
        """The slices of slit `slit_idx` in local axis"""
        
        # alpha and beta slices for slit `slit_idx`
        slices = self.slit_local_fov(slit_idx).to_slices(
            self.local_alpha_axis, self.local_beta_axis
        )
        # If slice to long, remove one pixel at the beginning or the end
        if (slices[1].stop - slices[1].start) > self.npix_slit_beta_width:
            if abs(
                self.local_beta_axis[slices[1].stop]
                - self.slit_local_fov(slit_idx).beta_end
            ) > abs(
                self.local_beta_axis[slices[1].start]
                - self.slit_local_fov(slit_idx).beta_start
            ):
                slices = (slices[0], slice(slices[1].start, slices[1].stop - 1))
            else:
                slices = (slices[0], slice(slices[1].start + 1, slices[1].stop))


        # # TODO Fix here 
        # if (slices[0].stop - slices[0].start) > self.npix_slit_alpha_width:
        #     slices = (slice(slices[0].start, slices[0].stop - 1), slices[1])
        # elif (slices[0].stop - slices[0].start) < self.npix_slit_alpha_width:
        #     slices = (slice(slices[0].start - 2, slices[0].stop), slices[1])
        return slices


    def get_slit_weights(self, slit_idx: int, slices: Tuple[slice, slice]):
        """The weights of the slit `slit_idx` in local axis"""

        weights = self.fov_weight(
            self.slit_local_fov(slit_idx),
            slices,
            self.local_alpha_axis,
            self.local_beta_axis,
        )

        # If previous do not share a pixel
        if slit_idx > 0:
            if self.get_slit_slices(slit_idx - 1)[1].stop - 1 != slices[1].start:
                weights[:, 0] = 1

        # If next do not share a pixel
        if slit_idx < self.npix_slit_beta_width - 1:
            if slices[1].stop - 1 != self.get_slit_slices(slit_idx + 1)[1].start:
                weights[:, -1] = 1

        return weights[np.newaxis, ...]


    def slicing(self, gridded_img: array, slit_idx: int) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.get_slit_slices(slit_idx=slit_idx)
        weights = self.get_slit_weights(slit_idx=slit_idx, slices=slices)   
        return gridded_img[slices[0], slices[1]] * weights[0]

    def slicing_t(
        self,
        slit: array,
        slit_idx: int,
        local_shape: Tuple[int, int, int]
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        out = np.zeros(local_shape)
        slices = self.get_slit_slices(slit_idx)
        weights = self.get_slit_weights(slit_idx, slices)
        out[slices[0], slices[1]] = slit * weights
        return out

    def forward(self, x: array) -> array:
        out = np.zeros(self.slices_shape)
        # # C
        blurred_x = jax_utils.idft(jax_utils.dft(x) * self.sotf, self.ishape)

        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred_x, pointing) 

            SS_gridded = jax_utils.idft(
                jax_utils.dft_mult(gridded, self._otf_sr*self.decalf),
                self.local_im_shape,
            )

            for slit_idx in range(self.instr.n_slit):
                sliced = self.slicing(SS_gridded, slit_idx)
                out[p_idx, slit_idx] = np.sum(sliced[: self.slices_shape[2] * self.srf : self.srf], axis=1)
        return out.ravel()
    

    def adjoint(self, data: array) -> array:
        global_img = np.zeros(self.ishape)
        for p_idx, pointing in enumerate(self.pointings):
            local_img = np.zeros(self.local_im_shape)
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            np.reshape(data, 
                                       self.slices_shape)[p_idx, slit_idx],
                            axis=1,
                        ),
                        self.npix_slit_beta_width,
                        axis=1,
                    )
                blurred_t_sliced = np.zeros(self.get_slit_shape_t())
                blurred_t_sliced[: self.slices_shape[2] * self.srf : self.srf,:] = oversampled_sliced
                local_img += self.slicing_t(blurred_t_sliced, slit_idx, self.local_im_shape)
                
            sum_t_img = jax_utils.idft(jax_utils.dft(local_img) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)

            degridded = self.gridding_t(np.array(sum_t_img, dtype=np.float64), pointing)
            global_img += degridded

        blurred_t_img = jax_utils.idft(jax_utils.dft_mult(global_img, self.sotf.conj()), self.ishape)
        return blurred_t_img
    

    def data_to_img(self, data: array) -> array:
        print("CF DEBUG : self.npix_slit_beta_width= ", self.npix_slit_beta_width)
        global_img = np.zeros(self.ishape)
        cum_grid = np.zeros((len(self.pointings), self.ishape[0], self.ishape[1]))
        for p_idx, pointing in enumerate(self.pointings):
            local_img = np.zeros(self.local_im_shape)
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            np.reshape(data, 
                                       self.slices_shape)[p_idx, slit_idx],
                            axis=1,
                        ),
                        self.npix_slit_beta_width,
                        axis=1,
                    )/self.npix_slit_beta_width
                blurred_t_sliced = np.zeros(self.get_slit_shape_t())
                blurred_t_sliced[: self.slices_shape[2] * self.srf : self.srf,:] = oversampled_sliced
                local_img += self.slicing_t(blurred_t_sliced, slit_idx, self.local_im_shape)
                
            sum_t_img = jax_utils.idft(jax_utils.dft(local_img) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)

            sum_t_img = np.array(sum_t_img)
            sum_t_img[sum_t_img<1] = 0
            sum_t_img[:,5] = sum_t_img[:,6]
            sum_t_img[:,153] = sum_t_img[:,152]

            degridded = self.gridding_t(np.array(sum_t_img, dtype=np.float64), pointing)
            global_img += degridded
            cum_grid[p_idx] = degridded
        #     plt.figure()
        #     plt.imshow(np.rot90(np.fliplr(np.flip(degridded)),3))
        #     plt.title(f"{p_idx}")
        #     plt.colorbar()
        # plt.show()
        valid_counts = np.sum(cum_grid != 0, axis=0)
        # plt.figure()
        # plt.imshow(np.rot90(np.fliplr(valid_counts),1))
        sum_of_values = np.sum(cum_grid, axis=0)
        weighted_mean = np.divide(sum_of_values, valid_counts, where=valid_counts != 0)
        

        return weighted_mean, global_img


    def gridding(self, blurred_img: array, pointing: instru.Coord) -> array:

        def find_nearest_idx(array, value):
            array = np.asarray(array)
            return (np.abs(array - value)).argmin()
        

        alpha_width = len(self.local_alpha_axis)
        beta_width = len(self.local_beta_axis)
        idx_alpha = find_nearest_idx(self.alpha_axis, pointing.alpha)
        idx_beta = find_nearest_idx(self.beta_axis, pointing.beta)

        alpha_start = idx_alpha -alpha_width//2
        alpha_end = idx_alpha + alpha_width//2+1

        beta_start = idx_beta -beta_width//2
        beta_end = idx_beta + beta_width//2+1
        alpha_slice = slice(alpha_start, alpha_end, None)
        beta_slice = slice(beta_start, beta_end, None)


        return blurred_img[alpha_slice, beta_slice]

    def gridding_t(self, local_img: array, pointing: instru.Coord) -> array:

        def find_nearest_idx(array, value):
            array = np.asarray(array)
            return (np.abs(array - value)).argmin()
        

        alpha_width = len(self.local_alpha_axis)
        beta_width = len(self.local_beta_axis)
        idx_alpha = find_nearest_idx(self.alpha_axis, pointing.alpha)
        idx_beta = find_nearest_idx(self.beta_axis, pointing.beta)

        alpha_start = idx_alpha -alpha_width//2
        alpha_end = idx_alpha + alpha_width//2+1

        beta_start = idx_beta -beta_width//2
        beta_end = idx_beta + beta_width//2+1

        alpha_slice = slice(alpha_start, alpha_end, None)
        beta_slice = slice(beta_start, beta_end, None)

        global_img = np.zeros(self.ishape)
        global_img[alpha_slice, beta_slice]= local_img
        return global_img

    def project_FOV(self):

        plt.figure()
        alpha = [np.min(self.alpha_axis), np.max(self.alpha_axis), np.min(self.alpha_axis),np.max(self.alpha_axis)]
        beta = [np.min(self.beta_axis), np.min(self.beta_axis), np.max(self.beta_axis),np.max(self.beta_axis)]
        print(alpha, beta)
        plt.plot(alpha, beta, 'o', label='Reference')

        
        for p_idx, pointing in enumerate(self.pointings):
            f = self.instr.fov + pointing
            plt.plot(
                list(map(op.attrgetter("alpha"), f.vertices)) + [f.vertices[0].alpha],
                list(map(op.attrgetter("beta"), f.vertices)) + [f.vertices[0].beta],
                "-x",
                label=p_idx
            )
        plt.legend()
        plt.show()


    def fov_weight(
        self,
        fov: instru.LocalFOV,
        slices: Tuple[slice, slice],
        alpha_axis: array,
        beta_axis: array,
    ) -> array:
        """The weight windows of the FOV given slices of axis

        Notes
        -----
        Suppose the (floor, ceil) hypothesis of `LocalFOV.to_slices`.
        """
        alpha_step = alpha_axis[1] - alpha_axis[0]
        beta_step = beta_axis[1] - beta_axis[0]
        slice_alpha, slice_beta = slices

        selected_alpha = alpha_axis[slice_alpha]
        selected_beta = beta_axis[slice_beta]

        weights = np.ones(
            (slice_alpha.stop - slice_alpha.start, slice_beta.stop - slice_beta.start)
        )

        # Weight for first α for all β
        # weights[0, :] *= (
        #     wght := abs(selected_alpha[0] - alpha_step / 2 - fov.alpha_start) / alpha_step
        # )
        # assert (
        #     0 <= wght <= 1
        # ), f"Weight of first alpha observed pixel in slit must be in [0, 1] ({wght:.2f})"

        if selected_beta[0] - beta_step / 2 < fov.beta_start:
            weights[:, 0] = (
                wght := 1
                - abs(selected_beta[0] - beta_step / 2 - fov.beta_start) / beta_step
            )
            assert (
                0 <= wght <= 1
            ), f"Weight of first beta observed pixel in slit must be in [0, 1] ({wght:.2f})"

        # weights[-1, :] *= (
        #     wght := abs(selected_alpha[-1] + alpha_step / 2 - fov.alpha_end) / alpha_step
        # )
        # assert (
        #     0 <= wght <= 1
        # ), f"Weight of last alpha observed pixel in slit must be in [0, 1] ({wght:.2f})"

        if selected_beta[-1] + beta_step / 2 > fov.beta_end:
            weights[:, -1] = (
                wght := 1
                - abs(selected_beta[-1] + beta_step / 2 - fov.beta_end) / beta_step
            )
            assert (
                0 <= wght <= 1
            ), f"Weight of last beta observed pixel in slit must be in [0, 1] ({wght:.2f})"

        return weights