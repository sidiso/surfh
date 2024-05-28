from collections import namedtuple
from typing import Tuple
from math import ceil, floor

import numpy as np
from numpy import ndarray as array

from surfh.Models import instru
from surfh.ToolsDir import cython_2D_interpolation, matrix_op, jax_utils


# TODO Jax version : precompute slices 
# TODO Fix last colunm of beta not taking into account
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
    def slit_beta_width(self):
        """width of beta axis in slit local referential"""
        return self.instr.fov.beta_width/self.instr.n_slit
    
    @property
    def npix_slit_beta_width(self):
        """number of pixel for beta axis in slit local referential"""
        return int(ceil(self.slit_beta_width / (self.local_beta_axis[1] - self.local_beta_axis[0])))
    
    @property
    def slit_alpha_width(self):
        return self.instr.fov.alpha_width
    
    @property
    def npix_slit_alpha_width(self):
        step = self.local_alpha_axis[1] - self.local_alpha_axis[0]
        npix = int(ceil((self.slit_alpha_width / 2) / step)) - int(
            floor(-self.slit_alpha_width / 2 / step)
        )   
        if npix > len(self.local_alpha_axis):
            npix = npix - 1
        return npix
    

    def slicing(self, gridded_cube: array, slit_idx: int) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.get_slit_slices(slit_idx=slit_idx)
        weights = self.get_slit_weights(slit_idx=slit_idx, slices=slices)   
        return gridded_cube[:, slices[0], slices[1]] * weights

    # TODO Faut-il mettre les `weights` ici aussi ?
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
        out[:, slices[0], slices[1]] = slit * weights
        return out


    def slit_local_fov(self, slit_idx: int):
        """The FOV of slit `slit_idx` in local ref"""
        slit_fov = self.instr.slit_fov[slit_idx]
        return slit_fov.local + self.instr.slit_shift[slit_idx]
    
    def local_fov_to_slices(self, localFov: instru.LocalFOV):
        """Return slices of axis that contains the slice

        Parameters
        ----------
        localFov: localFOV
          localFOV object for a specific slit
        alpha_axis: array
          alpha in local referential
        beta_axis: array
          beta in local referential
        """
        alpha_step = self.local_alpha_axis[1] - self.local_alpha_axis[0]
        beta_step  = self.local_beta_axis[1] - self.local_beta_axis[0]

        alpha_slice = slice(
                np.flatnonzero(self.local_alpha_axis > localFov.alpha_start - alpha_step/2)[0],
                np.flatnonzero(self.local_alpha_axis < localFov.alpha_end + alpha_step/2)[-1] + 1 # The '+1' because slice reach 'end -1'

        )

        beta_slice = slice(
                np.flatnonzero(self.local_beta_axis > localFov.beta_start - beta_step/2)[0],
                np.flatnonzero(self.local_beta_axis < localFov.beta_end + beta_step/2)[-1] + 1# The '+1' because slice reach 'end -1'
        )
        return (alpha_slice, beta_slice,)
    

    def get_slit_slices(self, slit_idx: int) -> Tuple[slice, slice]:
        """The slices of slit `slit_idx` in local axis"""
        
        # alpha and beta slices for slit `slit_idx`
        slices = self.local_fov_to_slices(self.slit_local_fov(slit_idx))
        if (slices[1].stop - slices[1].start) > self.npix_slit_beta_width:
            #raise Exception(f"the number of pixel in beta slit is supposed to be {self.npix_slit_beta_width}, but is {slices[1].stop - slices[1].start}")
            if slices[1].stop < len(self.local_beta_axis):
                    
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
            else:
                slices = (slices[0], slice(slices[1].start, slices[1].stop - 1))

        if (slices[0].stop - slices[0].start) > self.npix_slit_alpha_width:
            slices = (slice(slices[0].start, slices[0].stop - 1), slices[1])
        elif (slices[0].stop - slices[0].start) < self.npix_slit_alpha_width:
            slices = (slice(slices[0].start, slices[0].stop + 1), slices[1])

        return slices


    def get_slit_weights(self, slit_idx: int, slices: Tuple[slice, slice]):
        """The weights of the slit `slit_idx` in local axis"""
        weights = matrix_op.fov_weight(
            self.slit_local_fov(slit_idx),
            slices,
            self.local_alpha_axis,
            self.local_beta_axis,
        )
        # If previous do not share a pixel
        if slit_idx == 0:
            weights[:, 0] = 1
        if slit_idx > 0:
            if self.get_slit_slices(slit_idx - 1)[1].stop - 1 != slices[1].start:
                weights[:, 0] = 1

        # If next do not share a pixel
        if slit_idx < self.npix_slit_beta_width - 1:
            if slices[1].stop - 1 != self.get_slit_slices(slit_idx + 1)[1].start:
                weights[:, -1] = 1

        if slit_idx == self.npix_slit_beta_width - 1:
            weights[:, -1] = 1

        return weights[np.newaxis, ...]
