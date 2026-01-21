#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:02:21 2025

@author: dpineau
"""

import numpy as np
from scipy.integrate import trapezoid

from udft import irdftn, rdft2, ir2fr

from aljabr import LinOp

from surfh.ToolsDir import jax_utils
# imager model for fusion with spectro using decimation di and dj

# PSFs are full and partitionned to allow the direct sum with the spectro hessian
class Mirim_Model_LMM(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, pixel_arcsec=0.111, precompute_H_freq=None, low_mem=True
    ):
        """

        Parameters
        ----------
        psfs_monoch : np.array
            les PSF monochromatiques, format (L, I, J).
        L_pce : np.array
            les PCE de l'imageur, format (9, L).
        lamb_cube : np.array
            les longueurs d'onde du cube reconstruit en µm, format (L).'
        L_specs : np.array
            les spectres du modèle de mélange, format (5, L).
        shape_target : tuple
            dimensions spatiales du cube reconstruit, format (I, J).
        pixel_arcsec : float
            FoV en arcsec recouverte par chaque pixel du cube reconstruit (= FoV pour chaque pixel imageur).
            Utilisée pour la conversion d'unité.
            Par défaut : 0.111 arcsec.

        """
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        # print("Use of class:", "Mirim_Model_Full_Part")
        n_spec, n_lamb = L_specs.shape
        self.n_spec = n_spec
        self.shape_target = shape_target
        self.L_specs = L_specs
        self.psfs_monoch = psfs_monoch
        self.lamb_cube = lamb_cube

        specs = L_specs[np.newaxis, :, :, np.newaxis, np.newaxis]  # (1, 5, 300, 1, 1)
        psfs = psfs_monoch[np.newaxis, np.newaxis, ...]  # (1, 1, 300, 250, 500)
        L_pce = self.unit_conversion(L_pce, lamb_cube * 1e-6, pixel_arcsec)
        pce = L_pce[:, np.newaxis, :, np.newaxis, np.newaxis]  # (9, 1, 300, 1, 1)
        
        # H_int = trapezoid(specs * psfs * pce, x=lamb_cube, axis=2)  # (9, 300, 250, 500)
        # TODO: new normalisation added here
        # pce_norms = np.sum(L_pce, axis=1)[:, np.newaxis, np.newaxis, np.newaxis]

        if not low_mem:      
            if precompute_H_freq is None:
                pce_norms = trapezoid(L_pce * lamb_cube[np.newaxis, ...], x = lamb_cube, axis = 1)[:, np.newaxis, np.newaxis, np.newaxis]
                new_lamb_cube = lamb_cube[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                H_int = trapezoid(specs * pce * psfs * new_lamb_cube, x=lamb_cube, axis=2) / pce_norms # (9, 5, 250, 500)
                
                H_freq = ir2fr(H_int, shape_target, real=True)
            else:
                H_freq = precompute_H_freq
            self.H_freq = H_freq
        else:
            if precompute_H_freq is None:
                # -----------------------------
                # Poids du trapèze (spectral)
                # -----------------------------
                dl = np.diff(lamb_cube)
                trap_weights = np.empty_like(lamb_cube)
                trap_weights[1:-1] = 0.5 * (dl[:-1] + dl[1:])
                trap_weights[0]    = 0.5 * dl[0]
                trap_weights[-1]   = 0.5 * dl[-1]

                # -----------------------------
                # Normalisation PCE (physique)
                # -----------------------------
                # ∫ PCE(λ) * λ dλ
                pce_norms = trapezoid(
                    L_pce * lamb_cube[np.newaxis, :],
                    x=lamb_cube,
                    axis=1
                )[:, None, None, None]   # (9,1,1,1)

                # -----------------------------
                # Poids spectraux
                # -----------------------------
                # weights[p,s,l] = PCE[p,l] * Spec[s,l] * λ[l] * w[l]
                weights = (
                    L_pce[:, None, :] *
                    self.L_specs[None, :, :] *
                    lamb_cube[None, None, :] *
                    trap_weights[None, None, :]
                )  # (9, n_spec, L)

                # -----------------------------
                # Intégration spectrale
                # -----------------------------
                # contraction sur l'axe λ
                H_int = np.tensordot(
                    weights,             # (9, n_spec, L)
                    psfs_monoch,         # (L, I, J)
                    axes=([2], [0])
                )  # (9, n_spec, I, J)

                # -----------------------------
                # Normalisation finale
                # -----------------------------
                H_int /= pce_norms

                # -----------------------------
                # Passage en domaine fréquentiel
                # -----------------------------
                H_freq = ir2fr(H_int, shape_target, real=True)
            else:
                H_freq = precompute_H_freq

            self.H_freq = H_freq

        n_bands, _ = L_pce.shape
        self.n_bands = n_bands
        
        self.L_pce = L_pce
        
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(n_spec, shape_target[0], shape_target[1]),
            oshape=(n_bands, shape_target[0], shape_target[1]),
        )

    def forward(self, x):  # shape of x: (5, 250, 500), costs 2
        return np.real(irdftn(np.sum(self.H_freq * rdft2(x)[np.newaxis, ...], axis=1), shape = self.shape_target))
    
    def test_forward(self, x):
        cube = jax_utils.lmm_maps2cube(x, self.L_specs)
        blurred_cube = jax_utils.idft(jax_utils.dft(cube) * self.psfs_monoch, (self.ishape[1], self.ishape[2]))

        y = np.zeros((self.n_bands, self.ishape[1], self.ishape[2]))
        for i in range(self.n_bands):
            tmp = blurred_cube*self.L_pce[i, :, np.newaxis, np.newaxis]
            print("shape ", self.L_pce[i, :, np.newaxis, np.newaxis].shape)
            y[i] = trapezoid(tmp, x=self.lamb_cube, axis=0)
        return y

    def adjoint(self, y):  # shape of y: (9, 250, 500)
        return np.real(irdftn(np.sum(np.conj(self.H_freq) * rdft2(y)[:, np.newaxis, ...], axis=0), shape = self.shape_target))

    def fwadj(self, point):
        """Apply `Aᴴ·A` operator."""
        return self.adjoint(self.forward(point))

    def mapsToCube(self, maps):
        from jax import numpy as jnp
        self.L_specs = self.L_specs[:, ::4]

        print(f"SHape maps in mapsToCube: {maps.shape}")
        print(f"Shape L_specs in mapsToCube: {self.L_specs.shape}")
        if maps.shape[1]<200:
            return jax_utils.lmm_maps2cube(maps, self.L_specs)
        else:
            batch_size = 50  # ajustable
            n_spec, H, W = maps.shape
            n_lamb = self.L_specs.shape[1]

            # sécurité
            assert n_spec == self.L_specs.shape[0], \
                f"maps.shape[0]={n_spec} doit être égal à L_specs.shape[0]={self.L_specs.shape[0]}"

            cube = np.zeros((n_lamb, H, W), dtype=maps.dtype)

            # traitement par batch
            for start in range(0, n_lamb, batch_size):
                end = min(start + batch_size, n_lamb)
                # slice batch spectral
                L_batch = self.L_specs[:, start:end]  # (n_spec, batch)
                # contraction : tensordot sur l'axe n_spec
                cube[start:end] = np.tensordot(L_batch, maps, axes=([0],[0]))  # (batch, H, W)

            return cube
        

    # microJansky/arcsec^2 to microJansky/arcsec^2
    def unit_conversion(self, flux, lamb, pixel_arcsec):
        print("conversion des unités non appliquée")
        return flux