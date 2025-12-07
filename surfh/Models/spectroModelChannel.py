import pytest
import numpy as np
from aljabr import LinOp, dottest
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.Models import slicer, metadataMRS
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils, nearest_neighbor_interpolation, matrix_op
from surfh.Others import interpolation, slicing
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
from scipy.interpolate import griddata

from line_profiler import profile

from numba import njit, prange

import time





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
        
        # TODO: Update Resolving power value regarding specific wavelength fot this band 
        resolving_power = metadataMRS.build_MRS_resolving_power(chan=instr.name, resolving_path='/home/nmonnier/Data/JWST/NGC_7023/Fusion/Resolving_Power/', wavelength=self.global_wavelength_axis[self.wslice])
        print("Checking things", resolving_power.shape)
        self.instr.w_blur.resolving_power = resolving_power

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
        self.wpsf = np.ascontiguousarray(self.wpsf)
        
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
        self.make_mask()

        self.list_gridding_indexes = []
        for p_idx, pointing in enumerate(self.pointings):
            local_alpha_coord, local_beta_coord = (
                self.instr.fov + pointing
            ).local2global(self.local_alpha_axis, self.local_beta_axis)

            coords = np.vstack([
                local_alpha_coord.ravel(),
                local_beta_coord.ravel()
            ]).T

            i0,i1,j0,j1,wa0,wa1,wb0,wb1 = interpolation.precompute_gridding_indices(
                self.alpha_axis,
                self.beta_axis,
                coords[:,0],
                coords[:,1]
            )
            self.list_gridding_indexes.append((i0,i1,j0,j1,wa0,wa1,wb0,wb1))


        self.list_gridding_t_indexes = []
        for p_idx, pointing in enumerate(self.pointings):
            alpha_coord, beta_coord = (self.instr.fov + pointing).global2local(
                    self.alpha_axis, self.beta_axis
            )

            coords = np.vstack([alpha_coord.ravel(), beta_coord.ravel()]).T
            i0, i1, j0, j1, wa0, wa1, wb0, wb1 = interpolation.precompute_gridding_t_indices(
                                                                self.local_alpha_axis,
                                                                self.local_beta_axis,
                                                                coords[:, 0],
                                                                coords[:, 1]
                                                                )
            self.list_gridding_t_indexes.append((i0, i1, j0, j1, wa0, wa1, wb0, wb1))




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


    def old_gridding(self, blurred_cube: array, pointing: instru.Coord) -> array:

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
    
    def gridding(self, blurred_cube: array, p_idx: int) -> array:
        i0, i1, j0, j1, wa0, wa1, wb0, wb1 = self.list_gridding_indexes[p_idx]

        gridded = interpolation.gridding_cube(
            np.array(blurred_cube).astype(np.float64),
            i0,i1,j0,j1,
            wa0,wa1,wb0,wb1
        )
        gridded = gridded.reshape(
            blurred_cube.shape[0],
            self.local_alpha_axis.size,
            self.local_beta_axis.size
            )
        return gridded



    def old_gridding_t(self, local_cube: array, pointing: instru.Coord) -> array:

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


    def gridding_t(self, local_cube: array, p_idx: int) -> array:

        i0, i1, j0, j1, wa0, wa1, wb0, wb1 = self.list_gridding_t_indexes[p_idx]

        global_cube = interpolation.gridding_t_cube(
            local_cube,
            i0, i1, j0, j1, wa0, wa1, wb0, wb1
        )     

        return global_cube.reshape(-1, self.imshape[0], self.imshape[1])



    def test_project_mrsFov_to_specroFoV(self, MRSdata, pointing: instru.Coord):
        alpha_grid, beta_grid = np.meshgrid(self.alpha_axis, self.beta_axis, indexing='ij')
        optimized_global_coords = np.stack([alpha_grid.ravel(), beta_grid.ravel()], axis=-1)  # shape (Nx*Ny, 2)

        X_mrs, Y_mrs = np.meshgrid(self.local_alpha_axis, self.local_beta_axis, indexing='ij')  # forme (Nxmrs, Nymrs)
        mrs_coords = np.stack([X_mrs.ravel(), Y_mrs.ravel()], axis=0)  # shape (2, N)

        # === 3. Appliquer rotation et translation ===
        theta = np.deg2rad(self.instr.fov.angle)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated_coords = R @ mrs_coords  # shape (2, N)
        rotated_coords[0, :] += pointing.alpha  # RA
        rotated_coords[1, :] += pointing.beta  # DEC
        rotated_coords = rotated_coords.T  # shape (N, 2)
        # === 4. Interpolation de mrsFoV sur la grille RA/DEC ===
        mrsFoV_resampled = griddata(
            points=rotated_coords,
            values=MRSdata.ravel(),
            xi=optimized_global_coords,
            method='linear',
            fill_value=0
        )

        # === 5. Reformater en image 2D ===
        mrsFoV_on_spectroFoV = mrsFoV_resampled.reshape((len(self.alpha_axis), len(self.beta_axis)))
        return mrsFoV_on_spectroFoV


    def project_MRS_on_global(self, MRSdata, 
                            RA0, DEC0, angle_deg,
                            RA_grid, DEC_grid,
                            method='linear', fill_value=0):
        """
        Projette une image MRSdata dans une grille globale RA/DEC.

        Paramètres
        ----------
        MRSdata : 2D array
            Image locale à projeter.
        local_alpha_axis : 1D array
            Coordonnées locales alpha (axe horizontal) en degrés.
        local_beta_axis : 1D array
            Coordonnées locales beta (axe vertical) en degrés.
        RA0, DEC0 : float
            Coordonnées du pointage (centre de l'image) en degrés.
        angle_deg : float
            Angle de rotation de l'image locale, en degrés.
        RA_grid, DEC_grid : 2D arrays
            Grille globale des coordonnées RA/DEC sur laquelle projeter.
        method : str
            Méthode d'interpolation ('linear', 'nearest', etc.).
        fill_value : float
            Valeur par défaut en dehors des zones interpolées.

        Retour
        ------
        projected_image : 2D array
            Image projetée dans le repère global.
        """
        print(f"RA0: {RA0}, DEC0: {DEC0}, angle_deg: {angle_deg}")
        # Grille locale
        X_local, Y_local = np.meshgrid(self.local_alpha_axis, self.local_beta_axis, indexing='ij')  # Shape (Ny, Nx)

        # Aplatir et empiler en coordonnées locales (alpha, beta)
        coords_local = np.stack([X_local.ravel(), Y_local.ravel()], axis=0)  # (2, N)

        # Appliquer la rotation
        theta = np.deg2rad(angle_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        coords_rotated = R @ coords_local  # (2, N)

        # Appliquer le pointage RA/DEC
        coords_rotated[0, :] += RA0
        coords_rotated[1, :] += DEC0

        # Préparer les points RA/DEC globaux pour l'interpolation
        coords_rotated = coords_rotated.T  # (N, 2)
        coords_target = np.stack([RA_grid.ravel(), DEC_grid.ravel()], axis=-1)  # (M, 2)
        
        print(f"mean RA_grid: {np.mean(RA_grid)}, mean DEC_grid: {np.mean(DEC_grid)}")

        # Interpolation
        values = MRSdata.ravel()
        interpolated = griddata(
            points=coords_rotated,
            values=values,
            xi=coords_target,
            method=method,
            fill_value=fill_value
        )

        # Reshape pour avoir une image
        projected_image = interpolated.reshape(RA_grid.shape)
        return projected_image



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

    @profile
    def forward(self, blurred_cube):
        chan_out = np.zeros(self.oshape)
        # test_chan_out = np.zeros(self.oshape)
        for p_idx, pointing in enumerate(self.pointings):
            # print(f"Instr {self.instr.name}, pointing {p_idx}")
            gridded = self.gridding(blurred_cube[self.wslice], p_idx) 

            sum_cube = jax_utils.idft(
                jax_utils.dft_mult(gridded, self._otf_sr*self.decalf),
                self.local_im_shape,
            )
                
            sliced_stack = np.stack([self.slicer.slicing(sum_cube, slit_idx) for slit_idx in range(self.instr.n_slit)], axis=0)

            sliced_dev = jax.device_put(sliced_stack)
            wpsf_dev = jax.device_put(self.wpsf)
            blurred_batch_dev = jax_utils.batched_wblur(sliced_dev, wpsf_dev)

            blurred_batch = np.asarray(jax.device_get(blurred_batch_dev))
            chan_out[p_idx] = blurred_batch[:,:, : self.oshape[3] * self.srf : self.srf]

        return chan_out.ravel()

    @profile
    def adjoint(self, inarray: np.ndarray) -> np.ndarray:

        inter_cube = np.zeros((self.wslice.stop-self.wslice.start, len(self.alpha_axis), len(self.beta_axis)))
        for p_idx, pointing in enumerate(self.pointings):
            local_cube = np.zeros((self.wslice.stop-self.wslice.start,
                                   self.local_im_shape[0],
                                   self.local_im_shape[1]))

            num_slits = self.instr.n_slit
            blurred_t_sliced = np.zeros((num_slits, *self.slicer.get_slit_shape_t()), dtype=np.float64)

            # --- Boucle JAX ---
            jax_results = []
            for slit_idx in range(num_slits):
                oversampled_sliced = np.repeat(
                    np.expand_dims(np.reshape(inarray, self.oshape)[p_idx, slit_idx], axis=2),
                    self.slicer.npix_slit_beta_width,
                    axis=2,
                )
                # Calcul JAX, pas encore converti en NumPy
                jax_out = jax_utils.wblur_t(oversampled_sliced, self.wpsf.conj())
                jax_results.append(jax_out)

            # --- Conversion NumPy en un seul bloc ---
            for idx, jax_out in enumerate(jax_results):
                # blurred_t_sliced[idx, :, :, :] = np.asarray(jax.device_get(jax_out))
                blurred_t_sliced[idx, :,: self.oshape[3] * self.srf : self.srf,:] = np.asarray(jax.device_get(jax_out))

            # --- Ensuite appel slicing_t_numba ou autre avec blurred_t_sliced déjà rempli ---
            for slit_idx in range(num_slits):
                ii, jj, weights = self.slicer.precomputed[slit_idx]
                slicing.slicing_t_numba(local_cube, blurred_t_sliced[slit_idx], ii, jj, weights)



            sum_t_cube = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), p_idx)

            # inter_cube += degridded
            matrix_op.add_cube(inter_cube, degridded)

        return inter_cube*self.chan_mask[np.newaxis,...]

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

        degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), 0)
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


    def multi_dith_sliceToCube(self, inarray, ndith=[0]):
        global_img = np.zeros( (self.wslice.stop-self.wslice.start, self.imshape[0], self.imshape[1]))
        cg2 = []
        
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
                    )/(self.slicer.npix_slit_beta_width * self.srf)
                
                blurred_t_sliced = np.zeros(self.slicer.get_slit_shape_t())
                blurred_t_sliced[:,: self.oshape[3] * self.srf : self.srf,:] = jax_utils.wblur_t(oversampled_sliced.astype(np.float32), self.wpsf_dirac[:,:,:].conj())
             
             
                local_cube += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (self.wslice.stop-self.wslice.start, 
                                                                                 self.local_im_shape[0],
                                                                                 self.local_im_shape[1]))
                
            sum_t_img = jax_utils.idft(jax_utils.dft(local_cube) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)

            sum_t_img = np.array(sum_t_img)
            sum_t_img[sum_t_img<1] = 0

            degridded = self.gridding_t(np.array(sum_t_img, dtype=np.float64), p_idx)
            global_img += degridded

            cg2.append(np.ma.masked_less(degridded, 1))

        return np.ma.mean(cg2, axis=0)

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

            degridded = self.gridding_t(np.array(sum_t_cube, dtype=np.float64), p_idx)
            inter_cube[self.wslice, ...] += degridded
        return inter_cube



    def make_mask(self):
        """
        Make one mask per channel.
        """
        nslice = 50
        global_img = np.zeros(self.imshape)
        cum_grid = np.zeros((len(self.pointings), self.imshape[0], self.imshape[1]))

        # Select data for specific wavelength
        data = np.ones((self.oshape[0], self.oshape[1], 1, self.oshape[3])).ravel() * 1000

        for p_idx, pointing in enumerate(self.pointings):
            local_img = np.zeros(self.local_im_shape)
            for slit_idx in range(self.instr.n_slit):
                oversampled_sliced = np.repeat(
                        np.expand_dims(
                            np.reshape(data, 
                                    self.slices_shape)[p_idx, slit_idx],
                            axis=1,
                        ),
                        self.slicer.npix_slit_beta_width,
                        axis=1,
                    )/(self.slicer.npix_slit_beta_width * self.srf)
                blurred_t_sliced = np.zeros((1, self.slicer.get_slit_shape_t()[1], self.slicer.get_slit_shape_t()[2]))
                blurred_t_sliced[0,: self.slices_shape[2] * self.srf : self.srf,:] = oversampled_sliced
                local_img += self.slicer.slicing_t(blurred_t_sliced, slit_idx, (1, self.local_im_shape[0],self.local_im_shape[1]))[0]
                
            sum_t_img = jax_utils.idft(jax_utils.dft(local_img) * self._otf_sr.conj()*self.decalf.conj(), 
                                        self.local_im_shape)

            degridded = self.old_gridding_t(np.array(sum_t_img, dtype=np.float64), pointing)[0]
            global_img += degridded
            cum_grid[p_idx] = degridded

        binary_mask = global_img > 5
        self.chan_mask = binary_mask

