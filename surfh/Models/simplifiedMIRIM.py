import numpy as np
import itertools as it
from aljabr import LinOp
from scipy.integrate import trapezoid

from udft import irdftn, rdft2, ir2fr, dft2, idft2
from einops import einsum, rearrange

from surfh.Signalprocessing.utilities import partitioning_einops2


# # PSFs are full and partitionned to allow the direct sum with the spectro hessian
# class Mirim_Model_LMM(LinOp):
#     def __init__(
#         self, psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, precompute_H_freq=None
#     ):
#         """

#         Parameters
#         ----------
#         psfs_monoch : np.array
#             les PSF monochromatiques, format (L, I, J).
#         L_pce : np.array
#             les PCE de l'imageur, format (9, L).
#         lamb_cube : np.array
#             les longueurs d'onde du cube reconstruit en µm, format (L).'
#         L_specs : np.array
#             les spectres du modèle de mélange, format (5, L).
#         shape_target : tuple
#             dimensions spatiales du cube reconstruit, format (I, J).
#         pixel_arcsec : float
#             FoV en arcsec recouverte par chaque pixel du cube reconstruit (= FoV pour chaque pixel imageur).
#             Utilisée pour la conversion d'unité.
#             Par défaut : 0.111 arcsec.

#         """
#         assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
#         assert psfs_monoch.shape[2] <= shape_target[1]
        
#         # print("Use of class:", "Mirim_Model_Full_Part")
#         n_spec, n_lamb = L_specs.shape
#         self.n_spec = n_spec
#         self.shape_target = shape_target

#         specs = L_specs[np.newaxis, :, :, np.newaxis, np.newaxis]  # (1, 5, 300, 1, 1)
#         psfs = psfs_monoch[np.newaxis, np.newaxis, ...]  # (1, 1, 300, 250, 500)
#         pce = L_pce[:, np.newaxis, :, np.newaxis, np.newaxis]  # (9, 1, 300, 1, 1)
        
#         # H_int = trapezoid(specs * psfs * pce, x=lamb_cube, axis=2)  # (9, 300, 250, 500)
#         # TODO: new normalisation added here
#         # pce_norms = np.sum(L_pce, axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
#         if precompute_H_freq is None:
#             pce_norms = trapezoid(L_pce * lamb_cube[np.newaxis, ...], x = lamb_cube, axis = 1)[:, np.newaxis, np.newaxis, np.newaxis]
#             new_lamb_cube = lamb_cube[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
#             H_int = trapezoid(specs * psfs * pce * new_lamb_cube, x=lamb_cube, axis=2) / pce_norms # (9, 5, 250, 500)
            
#             H_freq = ir2fr(H_int, shape_target, real=True)
#         else:
#             H_freq = precompute_H_freq
#         self.H_freq = H_freq
        
#         n_bands, _ = L_pce.shape
#         self.n_bands = n_bands
        
#         self.L_pce = L_pce
        
#         self.psfs_monoch = psfs_monoch

#         super().__init__(
#             ishape=(n_spec, shape_target[0], shape_target[1]),
#             oshape=(n_bands, shape_target[0], shape_target[1]),
#         )

#     def forward(self, x):  # shape of x: (5, 250, 500), costs 2
#         return np.real(irdftn(np.sum(self.H_freq * rdft2(x)[np.newaxis, ...], axis=1), shape = self.shape_target))

#     def adjoint(self, y):  # shape of y: (9, 250, 500)
#         return np.real(irdftn(np.sum(np.conj(self.H_freq) * rdft2(y)[:, np.newaxis, ...], axis=0), shape = self.shape_target))

#     def fwadj(self, point):
#         """Apply `Aᴴ·A` operator."""
#         return self.adjoint(self.forward(point))



# PSFs are full and partitionned to allow the direct sum with the spectro hessian
class Mirim_Model_For_Fusion(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj, pixel_arcsec=0.111, H_int = None
    ):
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        # print("Use of class:", "Mirim_Model_Full_Part")
        n_spec, n_lamb = L_specs.shape
        self.n_spec = n_spec
        self.shape_target = shape_target

        specs = L_specs[np.newaxis, :, :, np.newaxis, np.newaxis]  # (1, 5, 300, 1, 1)
        psfs = psfs_monoch[np.newaxis, np.newaxis, ...]  # (1, 1, 300, 250, 500)
        L_pce = self.unit_conversion(L_pce, lamb_cube * 1e-6, pixel_arcsec)
        pce = L_pce[:, np.newaxis, :, np.newaxis, np.newaxis]  # (9, 1, 300, 1, 1)
        
        # H_int = trapezoid(specs * psfs * pce, x=lamb_cube, axis=2)  # (9, 300, 250, 500)
        # TODO: new normalisation added here
        # pce_norms = np.sum(L_pce, axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
        if H_int is None:
            pce_norms = trapezoid(L_pce * lamb_cube[np.newaxis, ...], x = lamb_cube, axis = 1)[:, np.newaxis, np.newaxis, np.newaxis]
            new_lamb_cube = lamb_cube[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            H_int = trapezoid(specs * psfs * pce * new_lamb_cube, x=lamb_cube, axis=2) / pce_norms 

        H_freq_full = ir2fr(H_int, shape_target, real=False)  # (9, 300, 250, 500)
        self.H_freq_full = H_freq_full
        
        H_freq = ir2fr(H_int, shape_target, real=True)
        self.H_freq = H_freq

        hess_mirim_freq_full = self.compute_hess_mirim(H_freq_full)
        part_hess_mirim_freq_full = self.partitioning_hess_mirim_freq_full2(hess_mirim_freq_full, di, dj)
        self.part_hess_mirim_freq_full = part_hess_mirim_freq_full  # (5, 5, 25, 25, 50, 100)

        n_bands, _ = L_pce.shape
        self.n_bands = n_bands
        
        self.di = di
        self.dj = dj
        
        self.L_pce = L_pce
        
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(n_spec, shape_target[0], shape_target[1]),
            oshape=(n_bands, shape_target[0], shape_target[1]),
        )

    def forward(self, x):  # shape of x: (5, 250, 500), costs 2
        return np.real(idft2(np.sum(self.H_freq_full * dft2(x)[np.newaxis, ...], axis=1)))
    
    def forward_freq_to_real(self, x_freq):  # shape of x: (5, 250, 500), input in freq, and output in real, costs 1
        return np.real(idft2(np.sum(self.H_freq_full * x_freq[np.newaxis, ...], axis=1)))
    
    def forward_freq_to_freq(self, x_freq):  # shape of x: (5, 250, 500), input and output in freq, costs 0
        return np.sum(self.H_freq_full * x_freq[np.newaxis, ...], axis=1)

    def adjoint(self, y):  # shape of y: (9, 250, 500)
        return np.real(idft2(np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)))
    
    def adjoint_real_to_freq_full(self, y):  # shape of y: (9, 250, 500)
        return np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)
    
    def adjoint_real_to_freq(self, y):  # shape of y: (9, 250, 500)
        return np.sum(np.conj(self.H_freq) * rdft2(y)[:, np.newaxis, ...], axis=0)
    
    # def adjoint_freq_full(self, y):  # shape of y: (9, 250, 500), return adjoint in fourier, and real = False
    #     return np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)

    def fwadj(self, x):  # shape of x: (5, 250, 500)
        return self.apply_hessian2(self.part_hess_mirim_freq_full, self.di, self.dj, self.shape_target, x)

    # microJansky/arcsec^2 to microJansky/arcsec^2
    def unit_conversion(self, flux, lamb, pixel_arcsec):
        print("conversion des unités non appliquée")
        return flux
    
    def compute_hess_mirim(self, H_freq):
        """compute H_freq^t * H_freq"""
        n_bands, n_spec, l, k = H_freq.shape
        hess_mirim = np.zeros(shape=(n_spec, n_spec, l, k), dtype=np.complex128)

        for p, q in it.product(range(n_spec), range(n_spec)):
            hess_mirim[p, q] = np.sum(np.conj(H_freq[:, p]) * H_freq[:, q], axis=0)

        return hess_mirim
    
    def partitioning_hess_mirim_freq_full2(self, hess_mirim_freq_full, di, dj):
        n_spec, _, H, W = hess_mirim_freq_full.shape
        h, w = int(H/di), int(W/dj)
        part_hess_mirim_freq_full = np.zeros(shape=(n_spec, n_spec, di * dj, di * dj, h, w), dtype=np.complex128)
        # (5, 5, 25, 25, 50, 100) comme hess_spec_freq

        for p, q in it.product(range(n_spec), range(n_spec)):
            hess_bloc = hess_mirim_freq_full[p, q]
            bloc_in_line = rearrange(
                hess_bloc, "(dx bx) (dy by) -> (dx dy) bx by", dx=di, dy=dj
            )
            for d in range(di * dj):
                part_hess_mirim_freq_full[p, q, d, d] += bloc_in_line[d]

        return part_hess_mirim_freq_full
    

    # compute the equivalent of a dot product between a matrix with a shape equal
    # to the hessian of the spectro (the hessian itself, its inverse, the hessian for fusion or its inverse) and an input with
    # a shape equal to abundance maps (abundance maps themselves or adjoint of data, or sum of adjoint data)
    # apply_hessian2, diff with apply_hessian: di and dj different, instead of di == dj
    def apply_hessian2(self, hess_spec_freq, di, dj, shape_target, x, x_is_freq_and_part=False):
        if x_is_freq_and_part:
            part_x_freq = x
        else:
            x_freq = dft2(x)

            # partitionnement de x
            part_x_freq = partitioning_einops2(x_freq, di, dj)  # (5, 25, 50, 100)
            # print("part_x_freq", part_x_freq.shape)

        # produit de HtH avec x
        HtH_x_freq = hess_spec_freq * part_x_freq[np.newaxis, :, np.newaxis, :, :, :]
        # (5, 5, 25, 25, 50, 100) * (1, 5, 1, 25, 50, 100) = (5, 5, 25, 25, 50, 100)
        # print("HtH_x_freq", HtH_x_freq.shape)

        # somme avec np.sum
        # HtH_x_freq_sum1 = np.sum(
        #     HtH_x_freq, axis=1
        # )  # (5, x, 25, 25, 50, 100) --> (5, 25, 25, 50, 100)
        # HtH_x_freq_sum2 = np.sum(
        #     HtH_x_freq_sum1, axis=2
        # )  # (5, 25, x, 50, 100) --> (5, 25, 50, 100)
        
        # # print("HtH_x_freq_sum2", HtH_x_freq_sum2.shape)

        # # # reconstitution des cartes en freq
        # # concat_HtH_x_freq = concatenating(HtH_x_freq_sum2, shape_target)[
        # #     :, :, : int(shape_target[1] / 2) + 1
        # # ]
        # # # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)

        # # # sortie de Fourier
        # # HtH_x = irdftn(concat_HtH_x_freq, shape_target)
        
        # # reconstitution des cartes en freq
        # concat_HtH_x_freq = concatenating2(HtH_x_freq_sum2, shape_target, di, dj)
        # # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)
        # # print("concat_HtH_x_freq", concat_HtH_x_freq.shape)
        
        # somme avec einsum
        HtH_x_freq_sum = einsum(HtH_x_freq, "ti tj di dj h w -> ti di h w")
        # reconstitution des cartes en freq
        concat_HtH_x_freq = self.concatenating2(HtH_x_freq_sum, shape_target, di, dj)

        # sortie de Fourier
        HtH_x = np.real(idft2(concat_HtH_x_freq))
        # print("HtH_x", HtH_x.shape)

        return HtH_x
    
    # diff with concatenating: now works with decim different for both dimensions
    def concatenating2(self, cubef, shape_target, di, dj):
        n_maps, d1_times_d2, h_block, w_block = cubef.shape
        h, w = shape_target
        
        concatenated_cube = np.zeros((n_maps, h, w), dtype=complex)
        k = 0
        for i in range(di):
            for j in range(dj):
                concatenated_cube[:, i * h_block : (i+1) * h_block, j * w_block : (j+1) * w_block] += cubef[:, k, :, :]
                k += 1
        
        return concatenated_cube
    
