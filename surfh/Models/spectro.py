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
from typing import List

import numpy as np
from numpy import ndarray as array
from aljabr import LinOp
from loguru import logger
import time
import psutil
import scipy as sp

from surfh.ToolsDir.utils import dft, idft
from surfh.Models import instru, channel


from surfh.Others import shared_dict
from surfh.Others.AsyncProcessPoolLight import APPL

import matplotlib.pyplot as plt

InputShape = namedtuple("InputShape", ["wavel", "alpha", "beta"])



class Spectro(LinOp):
    def __init__(
        self,
        instrs: List[instru.IFU],
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        sotf: array,
        pointings: instru.CoordList,
        verbose: bool = True,
        serial: bool = False,
    ):

        self.wavel_axis = wavel_axis
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis

        self.sotf = sotf
        self.pointings = pointings
        self.verbose = verbose
        self.serial = serial

        srfs = instru.get_srf(
            [chan.det_pix_size for chan in instrs],
            self.step,
        )

        _shared_metadata = shared_dict.create("s_metadata_" + str(time.time()))
        self._shared_metadata = _shared_metadata
        for instr in instrs:
            _shared_metadata.addSubdict(instr.get_name_pix())

        num_threads = np.ceil(psutil.cpu_count()/len(instrs))

        self.channels = [
            channel.Channel(
                instr,
                alpha_axis,
                beta_axis,
                wavel_axis,
                srf,
                pointings,
                _shared_metadata[instr.get_name_pix()].path,
                num_threads,
                self.serial,
            )
            for srf, instr in zip(srfs, instrs)
        ]

        self._idx = np.cumsum([0] + [np.prod(chan.oshape) for chan in self.channels])
        self.imshape = (len(alpha_axis), len(beta_axis))
        ishape = (len(wavel_axis), len(alpha_axis), len(beta_axis))
        oshape = (self._idx[-1],)

        super().__init__(ishape, oshape, "Spectro")
        self.check_observation()

    def __del__(self):
        """ Shut down all allocated memory e.g. shared arrays and dictionnaries"""
        if self._shared_metadata is not None:
            dico = shared_dict.attach(self._shared_metadata.path)
            dico.delete() 

    @property
    def step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    def get_chan_data(self, inarray: array, chan_idx: int) -> array:
        return np.reshape(
            inarray[self._idx[chan_idx] : self._idx[chan_idx + 1]],
            self.channels[chan_idx].oshape,
        )


    def forward(self, inarray: array) -> array:
        """
            MRS forward operator.
            Apply H

            inarray : 

            output :  
        """
        out = np.zeros(self.oshape)
        if self.verbose:
            logger.info(f"Spatial blurring DFT2({inarray.shape})")
        blurred_f = dft(inarray) * self.sotf
        for idx, chan in enumerate(self.channels):
            if self.verbose:
                logger.info(f"Channel {chan.name}")
            APPL.runJob("Forward_id:%d"%idx, chan.forward_multiproc_jax, 
                        args=(blurred_f,), 
                        serial=self.serial)
            
        APPL.awaitJobResult("Forward*", progress=self.verbose)
        
        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            out[self._idx[idx] : self._idx[idx + 1]] = fw_data.ravel()

        return out 

    
    def adjoint(self, inarray: array) -> array:
        """
            MRS adjoint operator.
            Apply H^t 

            inarray : 

            output :  
        """
        tmp = np.zeros(
            self.ishape[:2] + (self.ishape[2] // 2 + 1,), dtype=np.complex128
        )
        for idx, chan in enumerate(self.channels):
            if self.verbose:
                logger.info(f"Channel {chan.name}")
            APPL.runJob("Adjoint_id:%d"%idx, chan.adjoint_multiproc, 
                        args=(np.reshape(inarray[self._idx[idx] : self._idx[idx + 1]], chan.oshape),), 
                        serial=self.serial)

        APPL.awaitJobResult("Adjoint*", progress=self.verbose)

        self._shared_metadata.reload()
        for idx, chan in enumerate(self.channels):
            ad_data = self._shared_metadata[chan.name]["ad_data"]
            tmp += ad_data
        
        if self.verbose:
            logger.info(f"Spatial blurring^T : IDFT2({tmp.shape})")
        return idft(tmp * self.sotf.conj(), self.imshape)


    def cubeToSlice(self, cube):
        """
            Convert hyperspectral cube into MRS data (in slices shape).
            Similar to forward operator, without spatial and spectral blurring. 

            cube : 
                hyperspectral cube. 

            output : 
                list of MRS slices.  
        """
        slices = []
        blurred_f = dft(cube)
        for idx, chan in enumerate(self.channels):
            
            slice = chan.cubeToSlice(blurred_f)
            slices.append(slice)
        return slices
    
    def realData_cubeToSlice(self, chan_name, cube):

        return 
    


    def sliceToCube(self, slices):
        """
            Convert MRS data (in slices shape) onto a list of hyperspectral cube.
            Similar to adjoint operator, without spatial and spectral blurring. 

            Slices : 
                MRS Forward data. 

            output : 
                list of hyperspectral cube. One cube per frequency band.  
        """
        tmp = np.zeros(
            self.ishape[:2] + (self.ishape[2] // 2 + 1,), dtype=np.complex128
        )
        
        for idx, chan in enumerate(self.channels):
            tmp += chan.sliceToCube(np.reshape(slices[self._idx[idx] : self._idx[idx + 1]], chan.oshape))

        return idft(tmp, self.imshape)


    def interpolate_FoV(self, cube: array, chan: channel.Channel) -> array:
        """
        Re-interpolate a cube using chan_x as reference coordinates.
        """
        return chan.gridding(cube, chan.pointings[0])


    def qdcoadd(self, measures: array) -> array:
        out = np.zeros(self.ishape)
        nhit = np.zeros(self.ishape)

        def interp_l(measures, i_wl, o_wl):
            slit_idx = np.arange(measures.shape[1])
            i_coord = (i_wl, slit_idx)
            o_coord = np.vstack(  # output coordinate
                [
                    np.tile(o_wl.reshape((-1, 1)), (1, len(slit_idx))).ravel(),
                    np.tile(slit_idx.reshape((1, -1)), (len(o_wl), 1)).ravel(),
                ]
            ).T
            return sp.interpolate.interpn(
                i_coord,  # input axis
                measures,
                o_coord,
                bounds_error=False,
                fill_value=0,
            ).reshape((len(o_wl), measures.shape[1]))

        def chan_coadd(measures, chan):
            out = np.zeros(self.ishape, dtype=np.float)
            for p_idx, pointing in enumerate(self.pointings):
                gridded = np.zeros(chan.local_shape)
                for slit_idx in range(chan.instr.n_slit):
                    sliced = np.zeros(chan.slit_shape(slit_idx))
                    # λ interpolation, α repeat, and β duplication (tiling)
                    sliced[:] = np.tile(
                        np.repeat(
                            interp_l(
                                measures[p_idx, slit_idx]
                                / chan.instr.pce[..., np.newaxis],
                                chan.instr.wavel_axis,
                                self.wavel_axis[chan.wslice],
                            ),
                            repeats=chan.srf,
                            axis=1,
                        )[..., np.newaxis][:, : sliced.shape[1], :],
                        reps=(1, 1, sliced.shape[2]),
                    )
                    gridded += chan.slicing_t(sliced, slit_idx)
                out[chan.wslice, ...] += chan.gridding_t(gridded, pointing)
            return out

        for idx, chan in enumerate(self.channels):
            out += chan_coadd(
                np.reshape(measures[self._idx[idx] : self._idx[idx + 1]], chan.oshape),
                chan,
            )
            nhit += chan_coadd(
                np.reshape(
                    np.ones_like(measures[self._idx[idx] : self._idx[idx + 1]]),
                    chan.oshape,
                ),
                chan,
            )
        return out / nhit, nhit
    


    def check_observation(self):
        """ Check if channels FoV for all pointing match the observed image FoV"""

        # Get the coordinates of the observed object 
        grid = (self.alpha_axis, self.beta_axis)

        for idx, chan in enumerate(self.channels):
            # Get local alpha and beta coordinates for the channel
            local_alpha_axis = shared_dict.attach(chan._metadata_path)["local_alpha_axis"]
            local_beta_axis = shared_dict.attach(chan._metadata_path)["local_beta_axis"]
            
            for p_idx, pointing in enumerate(chan.pointings):
                out_of_bound = False
                # Get the global alpha and beta coordinates regarding the pointing for specific IFU
                alpha_coord, beta_coord = (chan.instr.fov + pointing).local2global(
                    local_alpha_axis, local_beta_axis
                )
                local_coords = np.vstack(
                            [
                                alpha_coord.ravel(),
                                beta_coord.ravel()
                            ]
                        ).T  
                
                # Check if IFU FoV anf image FoV match
                for i, p in enumerate(local_coords.T):
                    if not np.logical_and(np.all(grid[i][0] <= p),
                                        np.all(p <= grid[i][-1])):
                        out_of_bound = True

                if out_of_bound:
                    logger.debug(f"Out of bound for Chan {chan.name} - Pointing n°{p_idx}")

    def get_slits_data(self):
        """
        Return result of Forward operator in the right shape 
        [band, obs, slices, lambda, alpha]
        """
        slits = []
        for idx, chan in enumerate(self.channels):
            fw_data = self._shared_metadata[chan.name]["fw_data"]
            slits.append(fw_data)

        return slits

    def plot_slits_data(self, chan: int, obs : int):
        """
        Plot the data of each slit generated by the Forward operator for
        a specific band and a specific observation (one dithering). 
        """
        ifu = self.channels[chan]
        slices_data = self._shared_metadata[ifu.name]["fw_data"]

        ifu_metadata = shared_dict.attach(ifu._metadata_path)

        columns = ifu.instr.n_slit
        fig, axs = plt.subplots(nrows=1, ncols=columns, figsize=(8, 8), tight_layout=True)

        axs[0].imshow(slices_data[obs,0,:,:])
        #axs[0].set_yticks(ifu.instr.wavel_axis[::23])
        nx = ifu.instr.wavel_axis.shape[0]
        no_labels = 7 # how many labels to see on axis x
        step_x = int(nx / (no_labels - 1)) # step between consecutive labels
        x_positions = np.arange(0,nx,step_x)
        x_labels = np.around(ifu.instr.wavel_axis[::step_x], 2)
        axs[0].set_yticks(x_positions, x_labels)

        for i in range(1, columns):
            axs[i].imshow(slices_data[obs,i, :,:])
            axs[i].axis("off")
        plt.show()


    def plot_local_projected_data(self, chan: int, obs : int, freq : int = 0):
        """
        Plot the projection of each slit, for a specific band and observation, 
        onto a local cube.
        """
        ifu = self.channels[chan]
        out = shared_dict.attach(ifu._metadata_path)["fw_data"]

        local_cube = np.zeros((201, 17, 21)) # Shape [Lamda, alpha, beta] --> beta = n_slit, alpha = alpha_resolution

        test = out[obs, :, :, :]
        test = test[..., np.newaxis]

        for l in range(test.shape[1]):
            for s in range(test.shape[0]):
                for a in range(test.shape[2]):
                    local_cube[l, a, s] = test[s, l, a, 0]

        plt.imshow(local_cube[freq, :, :])
        plt.legend()
        plt.show()

    def close(self):
        """ Shut down all allocated memory e.g. shared arrays and dictionnaries"""
        if self._shared_metadata is not None:
            dico = shared_dict.attach(self._shared_metadata.path)
            dico.delete() 
