# SURFH - SUper Resolution and Fusion for Hyperspectral images
#
# Copyright (C) 2022 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

from collections import namedtuple
from typing import Tuple

import numpy as np
from numpy import ndarray as array
from aljabr import LinOp
from loguru import logger
from math import ceil
import udft

from surfh.ToolsDir.utils import dft, idft
from surfh.Models import instru

from surfh.Others import shared_dict
from surfh.Others.AsyncProcessPoolLight import APPL
from surfh.ToolsDir import cython_2D_interpolation, matrix_op, jax_utils

import matplotlib.pyplot as plt

import jax.numpy as jnp

InputShape = namedtuple("InputShape", ["wavel", "alpha", "beta"])



class Channel(LinOp):
    """A channel with FOV, slit, spectral blurring and pce

    Attributs
    ---------
    instr: IFU
      The IFU that contains physical information.
    alpha_axis: array
      The alpha axis of the input.
    beta_axis: array
      The beta axis of the input.
    wavel_axis: array
      The wavelength axis of the input.
    srf: int
      The super resolution factor.
    pointings: `CoordList`
      The list of pointed coordinates.
    name: str
      The same name than `instr`.
    step: float
      The alpha step of alpha_axis
    wslice: slice
      The wavelength slice of input that match instr with 0.1 μm of margin.
    npix_slit: int
      The number of beta pixel inside a slit (across slit dim).
    n_alpha: int
      The number of input pixel inside a slit (along slit dim)
    local_alpha_axis, self.local_beta_axis: array
      The alpha and beta axis in local referential.
    ishape: tuple of int
      The input shape.
    oshape: tuple of int
      The output shape.
    imshape: tuple of int
      The image shape (without wavelength).
    cshape: tuplee of int
      The input cube shape (after wslice).
    local_shape: tuple of int
      The input cube shape in local referential.
    num_threads : int
      Number of threads used for parallel computation inside Channel's methods.
    """

    def __init__(
        self,
        instr: instru.IFU,
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        srf: int,
        pointings: instru.CoordList,
        shared_metadata_path: str,
        num_threads: int,
        serial: bool,
    ):
        """Forward model of a Channel

        Attributs
        ---------
        instr: IFU
          The IFU that contains physical information.
        alpha_axis: array
          The alpha axis of the input.
        beta_axis: array
          The beta axis of the input.
        wavel_axis: array
          The wavelength axis of the input.
        srf: int
          The super resolution factor.
        pointings: `CoordList`
          The list of pointed coordinates.
        num_threads : int
          Number of threads for multiproc parallelissation.
        Notes
        -----
        alpha and beta axis must have the same step and must be regular. This is
        not the case for wavel_axis that must only have incrising values.
        """

        self.old_tmp = None

        _metadata = shared_dict.attach(shared_metadata_path)
        self._metadata_path = _metadata.path
        _metadata["wavel_axis"] = wavel_axis
        _metadata["alpha_axis"] = alpha_axis
        _metadata["beta_axis"] = beta_axis       

        if not np.allclose(alpha_axis[1] - alpha_axis[0], beta_axis[1] - beta_axis[0]):
            logger.warning(
                "α and β step for input axis must be equals. Here α={da} and β={db}",
            )

        self.pointings = pointings.pix(self.step)
        self.instr = instr.pix(self.step)

        self.num_threads = num_threads
        self.serial = serial

        self.srf = srf
        self.imshape = (len(alpha_axis), len(beta_axis))

        _metadata["_otf_sr"] = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]
        self._otf_sr = udft.ir2fr(np.ones((srf, 1)), self.imshape)[np.newaxis, ...]

        _metadata["local_alpha_axis"], _metadata["local_beta_axis"] = self.instr.fov.local_coords(
            self.step,
            alpha_margin=5 * self.step,
            beta_margin=5 * self.step,
        )

        ishape = (len(wavel_axis), self.imshape[0], self.imshape[1] // 2 + 1)
        oshape = (
            len(self.pointings),
            self.instr.n_slit,
            self.instr.n_wavel,
            ceil(self.n_alpha / self.srf),  # self.n_alpha // self.srf,
        )
        self.cshape = (
            self.wslice.stop - self.wslice.start,
            len(alpha_axis),
            len(beta_axis),
        )
        self.local_shape = (
            # self.instr.n_wavel,
            self.wslice.stop - self.wslice.start,
            len(_metadata["local_alpha_axis"]),
            len(_metadata["local_beta_axis"]),
        )

        _metadata["fw_data"] = np.zeros(oshape)
        _metadata["ad_data"] = np.zeros(ishape, dtype=np.complex128)
 
        super().__init__(ishape, oshape, self.instr.name)

        self.save_memory = False
        if not self.save_memory:
            self.precompute_wpsf()

    @property
    def step(self) -> float:
        alpha_axis = shared_dict.attach(self._metadata_path)["alpha_axis"]
        return alpha_axis[1] - alpha_axis[0]

    @property
    def beta_step(self) -> float:
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]
        return beta_axis[1] - beta_axis[0]

    @property
    def n_alpha(self) -> int:
        """The number of input pixel inside a slit (along slit dim)"""
        return self.instr.fov.local.n_alpha(self.step)

    @property
    def wslice(self) -> slice:
        """The wavelength slice of input that match instr with 0.1 μm of margin."""
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        return self.instr.wslice(wavel_axis, 0.1)

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]
        return int(
            ceil(self.instr.slit_beta_width / (beta_axis[1] - beta_axis[0]))
        )

    @property
    def wavel_axis(self) -> array:
        """ """
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        return wavel_axis
    
    @property
    def local_alpha_axis(self) -> array:
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        return local_alpha_axis
    
    @property
    def local_beta_axis(self) -> array:
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]
        return local_beta_axis
    
    @property
    def alpha_axis(self) -> array:
        alpha_axis = shared_dict.attach(self._metadata_path)["alpha_axis"]
        return alpha_axis
    
    @property
    def beta_axis(self) -> array:
        beta_axis = shared_dict.attach(self._metadata_path)["beta_axis"]
        return beta_axis




    def slit_local_fov(self, slit_idx) -> instru.LocalFOV:
        """The FOV of slit `slit_idx` in local ref"""
        slit_fov = self.instr.slit_fov[slit_idx]
        return slit_fov.local + self.instr.slit_shift[slit_idx]

    def slit_slices(self, slit_idx: int) -> Tuple[slice, slice]:
        """The slices of slit `slit_idx` in local axis"""
        local_alpha_axis = self.local_alpha_axis
        local_beta_axis  = self.local_beta_axis    
        slices = self.slit_local_fov(slit_idx).to_slices(
            local_alpha_axis, local_beta_axis
        )
        # If slice to long, remove one pixel at the beginning or the end
        if (slices[1].stop - slices[1].start) > self.npix_slit:
            if abs(
                local_beta_axis[slices[1].stop]
                - self.slit_local_fov(slit_idx).beta_end
            ) > abs(
                local_beta_axis[slices[1].start]
                - self.slit_local_fov(slit_idx).beta_start
            ):
                slices = (slices[0], slice(slices[1].start, slices[1].stop - 1))
            else:
                slices = (slices[0], slice(slices[1].start + 1, slices[1].stop))
        return slices

    def slit_shape(self, slit_idx: int) -> Tuple[int, int, int]:
        """The shape of slit `slit_idx` in local axis"""
        slices = self.slit_slices(slit_idx)
        return (
            self.wslice.stop - self.wslice.start,
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
        )

    def slit_weights(self, slit_idx: int) -> array:
        """The weights of slit `slit_idx` in local axis"""
        local_alpha_axis = self.local_alpha_axis
        local_beta_axis  = self.local_beta_axis 
        slices = self.slit_slices(slit_idx)

        weights = matrix_op.fov_weight(
            self.slit_local_fov(slit_idx),
            slices,
            local_alpha_axis,
            local_beta_axis,
        )

        # If previous do not share a pixel
        if slit_idx > 0:
            if self.slit_slices(slit_idx - 1)[1].stop - 1 != slices[1].start:
                weights[:, 0] = 1

        # If next do not share a pixel
        if slit_idx < self.npix_slit - 1:
            if slices[1].stop - 1 != self.slit_slices(slit_idx + 1)[1].start:
                weights[:, -1] = 1

        return weights[np.newaxis, ...]

    def slicing(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        return gridded[:, slices[0], slices[1]] * weights

    def slicing_cube2Fente(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        return gridded[:, slices[0], slices[1]] * weights


    def slicing_t(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        out = np.zeros(self.local_shape)
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        out[:, slices[0], slices[1]] = gridded * weights
        return out

    def slicing_Fente2Cube_t(
        self,
        gridded: array,
        slit_idx: int,
    ) -> array:
        """Return a weighted slice of gridded. `slit_idx` start at 0."""
        out = np.zeros(self.local_shape)
        slices = self.slit_slices(slit_idx)
        weights = self.slit_weights(slit_idx)
        out[:, slices[0], slices[1]] = gridded* weights
        return out
        
    def gridding(self, inarray: array, pointing: instru.Coord) -> array:
        """Returns interpolation of inarray in local referential"""
        # α and β inside the FOV shifted to pointing, in the global ref.
        local_alpha_axis = self.local_alpha_axis
        local_beta_axis  = self.local_beta_axis 
        alpha_axis       = self.alpha_axis
        beta_axis        = self.beta_axis

        alpha_coord, beta_coord = (self.instr.fov + pointing).local2global(
            local_alpha_axis, local_beta_axis
        )
        

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx),) + alpha_coord.shape
        local_coords = np.vstack(
            [
                alpha_coord.ravel(),
                beta_coord.ravel()
            ]
        ).T          
        return cython_2D_interpolation.interpn( (alpha_axis, beta_axis), 
                                              inarray, 
                                              local_coords, 
                                              len(wl_idx)).reshape(out_shape)
    

    def gridding_t(self, inarray: array, pointing: instru.Coord) -> array:
        """Returns interpolation of inarray in global referential"""
        # α and β inside the FOV shifted to pointing, in the global ref.
        local_alpha_axis = self.local_alpha_axis
        local_beta_axis  = self.local_beta_axis 
        alpha_axis       = self.alpha_axis
        beta_axis        = self.beta_axis

        alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
            alpha_axis, beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx), len(alpha_axis), len(beta_axis))

        global_coords = np.vstack(
            [
                alpha_coord.ravel(),
                beta_coord.ravel()
            ]
        ).T
        return cython_2D_interpolation.interpn( (local_alpha_axis, local_beta_axis), 
                                              inarray, 
                                              global_coords, 
                                              len(wl_idx),
                                              bounds_error=False, 
                                              fill_value=0,).reshape(out_shape)


    def sblur(self, inarray_f: array) -> array:
        """Return spatial blurring of inarray_f in Fourier space for SR"""
        _otf_sr = shared_dict.attach(self._metadata_path)["_otf_sr"]
        return idft(
            inarray_f * _otf_sr,
            self.imshape,
        )


    def fourier_duplicate_t(self, inarray: array) -> array:
        """Return spatial blurring transpose of inarray for SR. Returns in Fourier space.
        inarray :   5. | 0. | 0. | 0. | 0. | 7. | 0. | 0. | 0. | 0. | 2. | 0. | 0. | 0. | 0. 
        _otf_sr :   1. | 1. | 1. | 1. | 1.
        
        output  :   5. | 5. | 5. | 7. | 7. | 7. | 7. | 7. | 2. | 2. | 2. | 2. | 2. | 0. | 0.
        """
        #_otf_sr = shared_dict.attach(self._metadata_path)["_otf_sr"]
        _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.imshape)[np.newaxis, ...]
        return dft(inarray) * _otf_sr.conj()

    def _wpsf(self, length: int, step: float, slit_idx: int, type: str = 'mrs') -> array:
        """Return spectral PSF"""
        # ∈ [0, β_s]
        wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
        beta_in_slit = np.arange(0, length) * step

        if type == 'dirac':
            return self.instr.spectral_psf(
                            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
                            wavel_axis[self.wslice],
                            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
                            type='dirac',
                        )

        if self.save_memory:
            wpsf = self.instr.spectral_psf(
                            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
                            wavel_axis[self.wslice],
                            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
                            type='mrs',
                        )
        else:
            wpsf = shared_dict.attach(self._metadata_path)["wpsf"][slit_idx]
            
        return wpsf

    def wblur(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring of inarray"""
        return matrix_op.wblur(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx), self.num_threads if not self.serial else 1)

    def wdirac_blur(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring transpose of inarray using a dirac function.
           Only used to create generate cube from Forward data with applying Adjoint operator. """    
        return matrix_op.cubeToSlice(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx, 'dirac'), self.num_threads if not self.serial else 1)
   
    def wblur_t(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring transpose of inarray"""
        return matrix_op.wblur_t(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx), self.num_threads if not self.serial else 1)

    def wdirac_blur_t(self, inarray: array, slit_idx: int) -> array:
        """Returns spectral blurring transpose of inarray using a dirac function.
           Only used to create generate cube from Forward data with applying Adjoint operator. """    
        return matrix_op.sliceToCube_t(inarray, self._wpsf(inarray.shape[2], self.beta_step, slit_idx, 'dirac'), self.num_threads if not self.serial else 1)


    def forward(self, inarray_f):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.

        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = shared_dict.attach(self._metadata_path)["fw_data"]
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]

                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]* self.wblur(sliced).sum(axis=2)
               

    def forward_multiproc_jax(self, inarray_f):
        """inarray is supposed in global coordinate, spatially blurred and in Fourier space.
        Output is an array of shape (pointing, slit, wavelength, alpha)."""
        # [pointing, slit, λ', α]
        out = shared_dict.attach(self._metadata_path)["fw_data"]
        blurred = self.sblur(inarray_f[self.wslice, ...])
        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                sliced = self.slicing(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                out[p_idx, slit_idx, :, :] = self.instr.pce[
                    ..., np.newaxis
                ]*jax_utils.wblur(sliced, self._wpsf(sliced.shape[2], self.beta_step, slit_idx))
                
                

    def adjoint(self, measures):
        out = shared_dict.attach(self._metadata_path)["ad_data"]
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                tmp = np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx] * self.instr.pce[..., np.newaxis],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )

                tmp2 = self.wblur_t(tmp)

                sliced[:, : self.oshape[3] * self.srf : self.srf] = tmp2
                    
                gridded += self.slicing_t(sliced, slit_idx)
            blurred += self.gridding_t(gridded, pointing)
        
        out[self.wslice, ...] += self.fourier_duplicate_t(blurred)


    def adjoint_multiproc_jax(self, measures):
        # out = shared_dict.attach(self._metadata_path)["ad_data"]
        out = np.zeros(self.ishape, dtype=np.complex128)
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                tmp = np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx] * self.instr.pce[..., np.newaxis],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )
                
                sliced[:, : self.oshape[3] * self.srf : self.srf] = jax_utils.wblur_t(tmp, 
                                                                                      self._wpsf(tmp.shape[2], 
                                                                                                 self.beta_step, 
                                                                                                 slit_idx)
                                                                                      ) 
                
                slices = self.slit_slices(slit_idx)
                weights = self.slit_weights(slit_idx)
                gridded[:, slices[0], slices[1]] += sliced * weights
           
            _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.local_shape[1:])[np.newaxis, ...]
            tmp3 = dft(gridded) * _otf_sr.conj() 
            tmp4 = idft(tmp3, self.local_shape[1:])
            blurred += self.gridding_t(np.array(tmp4).astype(np.float64), pointing)

        
        out[self.wslice, ...] += np.array(dft(blurred))
        return out








    def cubeToSlice(self, cube):
        """cube is supposed in global coordinate in Fourier space for a specific Channel and band.
        slices is an array of shape (pointing, slit, wavelength, alpha).
        Reshape input cube into slices without spatial and spectral blurring 
        done in Forward operator.
        """
        # [pointing, slit, λ', α]
        slices = np.zeros(self.oshape)
        blurred = idft(cube[self.wslice, ...], self.imshape) # Replace sblur

        for p_idx, pointing in enumerate(self.pointings):
            gridded = self.gridding(blurred, pointing)
            for slit_idx in range(self.instr.n_slit):
                # Slicing, weighting and α subsampling for SR
                sliced = self.slicing_cube2Fente(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
                slices[p_idx, slit_idx, :, :] = self.wdirac_blur(sliced, slit_idx).sum(axis=2) #TODO Change: That
        return slices      

    def realData_cubeToSlice(self, cube):
        slices = np.zeros(self.oshape[1:]) # Remove pointing dimension
        gridded = self.gridding(cube, instru.Coord(0, 0))
        for slit_idx in range(self.instr.n_slit):
            sliced = self.slicing_cube2Fente(gridded, slit_idx)[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
            slices[slit_idx, :, :] = sliced.sum(axis=2) # Only sum on the Beta axis
        return slices      







    def sliceToCube(self, measures):
        out = np.zeros(self.ishape, dtype=np.complex128)
        blurred = np.zeros(self.cshape)
        for p_idx, pointing in enumerate(self.pointings):
            gridded = np.zeros(self.local_shape)
            for slit_idx in range(self.instr.n_slit):
                sliced = np.zeros(self.slit_shape(slit_idx))
                # α zero-filling, λ blurrling_t, and β duplication
                tmp = np.repeat(
                        np.expand_dims(
                            measures[p_idx, slit_idx],
                            axis=2,
                        ),
                        sliced.shape[2],
                        axis=2,
                    )/sliced.shape[2] #TODO : Véririfier si diviser par la taille de beta est correct ?
                                      # Car dans le modèle direct/Adjoint c'est faux. Mais ici on fait que de la transformation
                                      # pas du modèle...
                tmp2 = self.wdirac_blur_t(tmp, slit_idx) # TODO fix wdirac_blur_t() function (it currently gived full 0)
                
                sliced[:, : self.oshape[3] * self.srf : self.srf] = tmp2

                out_slice = np.zeros((self.local_shape[0], self.local_shape[1], self.local_shape[2]))
                nslices = self.slit_slices(slit_idx)
                weights = self.slit_weights(slit_idx)
                print(tmp.shape)
                out_slice[:, nslices[0], nslices[1]] = sliced* weights



                gridded += out_slice#self.slicing_Fente2Cube_t(sliced, slit_idx)



                plt.show()

            _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.local_shape[1:])[np.newaxis, ...]
            tmp3 = dft(gridded) * _otf_sr.conj()
            tmp4 = idft(tmp3, self.local_shape[1:])

            blurred += self.gridding_t(tmp4, instru.Coord(0, 0))
            # blurred += self.gridding_t(gridded, pointing)
        out[self.wslice, ...] += dft(blurred)
        # out[self.wslice, ...] = self.fourier_duplicate_t(blurred)
        return out

    def realData_sliceToCube(self, slices, cube_dim):
        out = np.zeros(cube_dim)
        blurred = np.zeros(cube_dim)
        gridded = np.zeros((cube_dim[0] , self.local_shape[1], self.local_shape[2]))
        for slit_idx in range(self.instr.n_slit):
            dlt = self.slit_slices(slit_idx)
            sliced = np.zeros((cube_dim[0], dlt[0].stop - dlt[0].start, dlt[1].stop - dlt[1].start,))
            tmp = np.repeat(
                            np.expand_dims(
                                slices[slit_idx],
                                axis=2,
                                ),
                                sliced.shape[2],
                                axis=2,
                            )/sliced.shape[2]
            tmp2 = tmp
            sliced[:, : cube_dim[0] * self.srf : self.srf] = tmp2
            # Replace slicing_Fente2Cube_t to match the right shape
            out_slice = np.zeros((cube_dim[0], self.local_shape[1], self.local_shape[2]))
            nslices = self.slit_slices(slit_idx)
            weights = self.slit_weights(slit_idx)
            out_slice[:, nslices[0], nslices[1]] = sliced* weights
            gridded += out_slice

        _otf_sr = udft.ir2fr(np.ones((self.srf, 1)), self.local_shape[1:])[np.newaxis, ...]
        tmp = dft(gridded) * _otf_sr.conj() 

        # blurred += self.gridding_t(gridded, instru.Coord(0, 0))
        tmp2 = idft(tmp, self.local_shape[1:])
        
        blurred += self.gridding_t(tmp2, instru.Coord(0, 0))

        # Replace Fourier dupplicate to match the right shape
        # _otf_sr = udft.ir2fr(np.ones((1, self.srf)), cube_dim[1:])[np.newaxis, ...]
        # out = dft(blurred) * _otf_sr.conj() 
        # return idft(out, cube_dim[1:])
        return blurred

    def precompute_wpsf(self):
        local_alpha_axis = shared_dict.attach(self._metadata_path)["local_alpha_axis"]
        local_beta_axis = shared_dict.attach(self._metadata_path)["local_beta_axis"]

        _metadata = shared_dict.attach(self._metadata_path)
        _metadata["wpsf"] = {}
        alpha_coord, beta_coord = (self.instr.fov).local2global(
            local_alpha_axis, local_beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange((self.wslice.stop-self.wslice.start))

        out_shape = (len(wl_idx),) + alpha_coord.shape
        gridded = np.ones(out_shape)
       
        for slit_idx in range(self.instr.n_slit):
            slices = self.slit_slices(slit_idx)
            sliced = gridded[
                    :, slices[0], slices[1]
                ]
            sliced = sliced[
                    :, : self.oshape[3] * self.srf : self.srf
                ]
            
            wavel_axis = shared_dict.attach(self._metadata_path)["wavel_axis"]
            beta_in_slit = np.arange(0, sliced.shape[2]) * self.beta_step
            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
            wpsf = self.instr.w_blur.psfs(self.instr.wavel_axis, beta_in_slit - np.mean(beta_in_slit), wavel_axis[self.wslice], arcsec2micron)
            
            _metadata["wpsf"][slit_idx] = wpsf

