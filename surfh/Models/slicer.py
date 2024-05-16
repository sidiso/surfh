from collections import namedtuple
from typing import Tuple
from math import ceil

import numpy as np
from numpy import ndarray as array

from surfh.Models import instru
from surfh.ToolsDir import cython_2D_interpolation, matrix_op, jax_utils


# TODO Jax version : precompute slices 
class Slicer():

    def __init__(self,
                 instr: instru.IFU,
                 wavelength_axis: array,
                 alpha_axis: array,
                 beta_axis: array,
                 local_alpha_axis: array,
                 local_beta_axis: array):
        
        self.instr = instr
        self.wavelength_axis = wavelength_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        return int(
            ceil(self.instr.slit_beta_width / (self.beta_axis[1] - self.beta_axis[0]))
        )


    def slicing(self, gridded_cube: array, slit_idx: int) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.get_slit_slices(slit_idx=slit_idx)
        weights = self.get_slit_weights(slit_idx=slit_idx, slices=slices)        
        return gridded_cube[:, slices[0], slices[1]] * weights

    def slicing_t(
        self,
        gridded: array,
        slit_idx: int,
        local_shape: Tuple[int, int, int]
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        out = np.zeros(local_shape)
        slices = self.get_slit_weights(slit_idx)
        weights = self.get_slit_weights(slit_idx, slices)
        out[:, slices[0], slices[1]] = gridded * weights
        return out

    def get_slit_weights(self, slit_idx: int, slices: Tuple[slice, slice]):
        """The weights of the slit `slit_idx` in local axis"""
        weights = matrix_op.fov_weight(
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
        if slit_idx < self.npix_slit - 1:
            if slices[1].stop - 1 != self.get_slit_slices(slit_idx + 1)[1].start:
                weights[:, -1] = 1

        return weights[np.newaxis, ...]


    def get_slit_slices(self, slit_idx: int) -> Tuple[slice, slice]:
        """The slices of slit `slit_idx` in local axis"""
        slices = self.slit_local_fov(slit_idx).to_slices(
            self.local_alpha_axis, self.local_beta_axis
        )
        if (slices[1].stop - slices[1].start) > self.npix_slit:
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
        
        return slices


    def slit_local_fov(self, slit_idx: int):
        """The FOV of slit `slit_idx` in local ref"""
        slit_fov = self.instr.slit_fov[slit_idx]
        return slit_fov.local + self.instr.slit_shift[slit_idx]


    def slit_shape(self, slit_idx: int):
        """The shape of slit `slit_idx` in local axis"""
        slices = self.get_slit_slices(slit_idx)
        return (
            self.wslice.stop - self.wslice.start,
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
        )