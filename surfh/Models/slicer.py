from collections import namedtuple
from typing import Tuple
from math import ceil, floor

import numpy as np
from numpy import ndarray as array

from surfh.Models import instru
from surfh.ToolsDir import cython_2D_interpolation, matrix_op, jax_utils
from surfh.Others import slicing

from line_profiler import profile
# TODO Jax version : precompute slices 
# TODO Fix last colunm of beta not taking into account
class Slicer():

    def __init__(self,
                 instr: instru.IFU,
                 wavelength_axis: array,
                 alpha_axis: array,
                 beta_axis: array,
                 local_alpha_axis: array,
                 local_beta_axis: array,
                 srf: int):
        
        self.instr = instr
        self.wavelength_axis = wavelength_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.local_alpha_axis = local_alpha_axis
        self.local_beta_axis = local_beta_axis
        self.srf = srf
        self.slices_shape = (self.instr.n_slit, ceil(self.npix_slit_alpha_width / self.srf))

        self.slit_flat_i = []
        self.slit_flat_j = []
        self.slit_flat_w = []
        self.slit_size = []
        self.precompute_slices_weights()


    @property
    def wslice(self) -> slice:
        """The wavelength slice of input that match instr with 0.1 μm of margin."""
        return self.instr.wslice(self.wavelength_axis, 0.1)

    @property
    def slit_beta_width(self):
        """width of beta axis in slit local referential"""
        return self.instr.fov.beta_width/self.instr.n_slit
    
    @property
    def npix_slit_beta_width(self):
        """number of pixel for beta axis in slit local referential"""
        return int(ceil(self.slit_beta_width / (self.beta_axis[1] - self.beta_axis[0])))
    
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

    @profile
    def slicing(self, gridded_cube: array, slit_idx: int) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.get_slit_slices(slit_idx=slit_idx)
        weights = self.get_slit_weights(slit_idx=slit_idx, slices=slices)   
        return gridded_cube[:, slices[0], slices[1]] * weights

    # TODO Faut-il mettre les `weights` ici aussi ?
    @profile
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
        # print("Weights = ", weights)
        # print(slices[0], slices[1])
        tmp = slit * weights
        out[:, slices[0], slices[1]] = tmp
        # if slit.shape[0] == 646:
        #     print("slit.shape:", slit.shape)
        #     print("slit.strides:", slit.strides)
        #     print("slit.flags:", slit.flags)

        #     print("weights.shape:", weights.shape)
        #     print("weights.strides:", weights.strides)
        #     print("weights.flags:", weights.flags)

        #     print("out_orig[:, slices[0], slices[1]].shape:", tmp.shape)


        return out
    
    def new_slicing_t(self, local_cube, slit, slit_idx):
        slicing.apply_slit_slice(local_cube,
                        slit.reshape(slit.shape[0], -1),
                        self.slit_flat_w[slit_idx],
                        self.slit_flat_i[slit_idx],
                        self.slit_flat_j[slit_idx])

    def slicing_t_fast_debug(self, local_cube, slit, slit_idx):
        slices = self.get_slit_slices(slit_idx)
        weights = self.get_slit_weights(slit_idx, slices)

        # multiplication
        tmp = slit * weights
        # extrait les slices
        ii = np.arange(slices[0].start, slices[0].stop)
        jj = np.arange(slices[1].start, slices[1].stop)
        # écriture dans le cube
        local_cube[:, ii[:, None], jj[None, :]] += tmp

        return local_cube


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
        beta_step = self.local_beta_axis[1] - self.local_beta_axis[0]
        return (
            slice(
                np.flatnonzero(localFov.alpha_start < self.local_alpha_axis + alpha_step / 2)[0],
                np.flatnonzero(self.local_alpha_axis - alpha_step / 2 < localFov.alpha_end)[-1] + 1,
            ),
            slice(
                np.flatnonzero(localFov.beta_start < self.local_beta_axis + beta_step / 2)[0],
                np.flatnonzero(self.local_beta_axis - beta_step / 2 < localFov.beta_end)[-1] + 1,
            ),
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

        if self.slices_shape[1]%2 == 0 and self.slices_shape[1] < 28:
            # TODO Fix here 
            if (slices[0].stop - slices[0].start) > self.npix_slit_alpha_width:
                slices = (slice(slices[0].start, slices[0].stop - 1), slices[1])
            elif (slices[0].stop - slices[0].start) < self.npix_slit_alpha_width:
                slices = (slice(slices[0].start - 2, slices[0].stop), slices[1])

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
        if slit_idx < self.slices_shape[0] - 1:
            if slices[1].stop - 1 != self.get_slit_slices(slit_idx + 1)[1].start:
                weights[:, -1] = 1

        return weights[np.newaxis, ...]


    def get_slit_shape(self):
        slices = self.get_slit_slices(0)
        return (
            self.wslice.stop - self.wslice.start,
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
        )
    
    def get_slit_shape_t(self):
        slices = self.get_slit_slices(0)
        return (
            self.wslice.stop - self.wslice.start,
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
        )
    
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
    
    def precompute_slices_weights(self):
        self.precomputed = []
        for slit_idx in range(self.slices_shape[0]):
            slices = self.get_slit_slices(slit_idx)
            weights = self.get_slit_weights(slit_idx, slices)
            ii = np.arange(slices[0].start, slices[0].stop)
            jj = np.arange(slices[1].start, slices[1].stop)
            self.precomputed.append((ii, jj, weights))

def test_slicing_equivalence(slicer, slit, slit_idx, local_imshape):
    """
    slicer : ton objet Slicer
    slit   : le buffer 'blurred_t_sliced' exactement comme dans ton pipeline
    """
    # Taille locale
    L = slit.shape[0]
    local_shape = (L, local_imshape[0], local_imshape[1])

    # Ancienne version
    out_ref = slicer.slicing_t(slit, slit_idx, local_shape)

    # Nouvelle version
    local_cube = np.zeros(local_shape, dtype=np.float64)
    # slicer.slicing_t_fast_debug(local_cube, slit, slit_idx, None)
    ii, jj, weights = slicer.precomputed[slit_idx]
    slicing.slicing_t_numba(local_cube, slit, ii, jj, weights)


    # Comparaison
    diff = local_cube - out_ref
    maxabs = np.max(np.abs(diff))
    print(f"Test for Slit idx {slit_idx}")
    print("max abs diff:", maxabs)

    if maxabs > 1e-12:
        idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        l,i,j = idx
        print("First mismatch at (l,i,j) =", idx)
        print("ref value:", out_ref[l,i,j])
        print("new value:", local_cube[l,i,j])
        print("difference:", diff[l,i,j])

    return maxabs, out_ref, local_cube
