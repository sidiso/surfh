import numpy as np
import os
import matplotlib.pyplot as plt
from aljabr import LinOp

from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir, dft2, dftn, idft2
from einops import einsum, rearrange, reduce, repeat


class Spectro_Model_Cube(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, di:int, dj:int, lamb_cube, L_specs, shape_target # size of pixels before integration and decimation of spectro
    ):
        assert shape_target[0] % di == 0
        assert shape_target[1] % dj == 0
        
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
        kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target, real=False)[np.newaxis, ...] # (1, 250, 500)
    
        psfs_freq = ir2fr(
            psfs_monoch * L_pce[:, np.newaxis, np.newaxis],
            shape=shape_target,
            real=False,
        ) * kernel_for_sum_freq
        
        # # translation dans Fourier pour sauvegarde de la convolution en haut à gauche
        # # MÉTHODE 1
        decal = np.zeros(shape_target)
        dsi = int((di-1)/2)
        dsj = int((dj-1)/2)
        # if ds != 0:
        #     ds = 0
        # print("dsi", dsi, "dsj", dsj)
        decal[- dsi, - dsj] = np.sqrt(shape_target[0] * shape_target[1]) # surement pour annuler les normalisations qui arrivent dans le passage de Fourier ?
        decalf = dft2(decal)
        self.decal =decal
        self.decalf = decalf
        h_block, w_block = int(shape_target[0] / di), int(shape_target[1] / dj)
        
        # partitionnement
        # part_psfs_freq_full = partitioning_einops2(psfs_freq, di, dj)
        part_psfs_freq_full = partitioning_einops2(psfs_freq * decalf, di, dj)
        # print("part_psfs_freq_full", part_psfs_freq_full.shape)

        # conjugué des psfs partitionnées
        conj_part_psfs_freq_full = np.conj(part_psfs_freq_full)
        # print("conj_part_psfs_freq_full", conj_part_psfs_freq_full.shape)

        # produit des psfs avec les conjuguées
        # (300, 1, 25, 50, 100) * (300, 25, 1, 50, 100) = (300, 25, 25, 50, 100)
        mat = (
            (1 / (di * dj))
            * part_psfs_freq_full[:, np.newaxis, ...]
            * conj_part_psfs_freq_full[:, :, np.newaxis, ...]
        )
        # print("mat", mat.shape)

        # création de HtH
        L_specs_converted = L_specs
        specs = L_specs_converted[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]  # (5, 300, 1, 1)
        # print("check1")
        n_spec = specs.shape[0]
        HtH_freq = np.zeros(
            (n_spec, n_spec, di * dj, di * dj, h_block, w_block), dtype=complex
        )
        # print("check2")
        for k1 in range(n_spec):
            for k2 in range(k1, n_spec):
                HtH_freq[k1, k2] += np.sum(specs[k1] * specs[k2] * mat, axis=0)

        # print("check3")
        # utilisation de la symétrie de HtH
        for k1 in range(n_spec):
            for k2 in range(k1):
                HtH_freq[k1, k2] += HtH_freq[k2, k1]

        self.hess_spec_freq = HtH_freq

        # H_spec_freq utile pour forward et adjoint
        # self.H_spec_freq = make_H_spec_freq_sum2(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # )
        
        # print("2 x H_spec_freq enlevés !!")
        
        self.H_spec_freq = make_H_spec_freq_sum2(
            psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        ) * rdft2(decal)[np.newaxis, np.newaxis, :, :]
        
        # utile pour forward_freq_to_freq et forward_freq_to_real
        self.H_spec_freq_full = make_H_spec_freq_sum_full(
            psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        ) * decalf[np.newaxis, np.newaxis, :, :]
        
        self.di = di
        self.dj = dj
        self.shape_target = shape_target
        self.n_lamb = lamb_cube.shape[0]
        self.n_spec = n_spec
        self.L_pce = L_pce
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(self.n_spec, shape_target[0], shape_target[1]),
            oshape=(self.n_lamb, shape_target[0] // di, shape_target[1] // dj),
        )

    def forward(self, x): # input and output in real, costs 1
        assert x.shape == self.ishape
        
        x_freq = rdft2(x)[:, np.newaxis, ...]  # (5, 1, 250, 251)
        H_spec_x_freq = np.sum(
            self.H_spec_freq * x_freq, axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = irdftn(H_spec_x_freq, self.shape_target)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube
    
    def forward_freq_to_freq(self, x_freq): # input and output in freq, costs 2
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return dft2(decimated_cube)
    
    def forward_freq_to_real(self, x_freq): # input in freq, output in real, costs 1
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube

    def adjoint(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y
        

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        maps = irdftn(H_spec_x_freq, self.shape_target)  # (5, 250, 500)
        
        return maps  # shape = 5, 250, 500
    
    def adjoint_real_to_freq_full(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq_full) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    def adjoint_real_to_freq(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    # def adjoint_freq_full(self, y): # return adjoint in fourier, and real = False
    #     assert y.shape == self.oshape
        
    #     # bourrage de zéros
    #     original_cube = np.zeros(
    #         (self.n_lamb, self.shape_target[0], self.shape_target[1])
    #     )  # (300, 250, 500)
    #     original_cube[:, :: self.di, :: self.dj] = y
        

    #     # make convolution with conjugated weighted psfs
    #     original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
    #     H_spec_x_freq = np.sum(
    #         np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
    #     )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
    #     return H_spec_x_freq  # shape = 5, 250, 500

    def fwadj(self, x):
        assert x.shape == self.ishape
        return apply_hessian2(self.hess_spec_freq, self.di, self.dj, self.shape_target, x)


# class Spectro_Model_Cube(LinOp):
#     def __init__(
#         self, psfs_monoch, L_pce, di:int, dj:int, L_specs, shape_target
#     ):
#         assert shape_target[0] % di == 0
#         assert shape_target[1] % dj == 0
        
#         assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
#         assert psfs_monoch.shape[2] <= shape_target[1]
        
#         kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
#         kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target, real=False)[np.newaxis, ...] # (1, 250, 500)
        
#         # # translation dans Fourier pour sauvegarde de la convolution en haut à gauche
#         decal = np.zeros(shape_target)
#         dsi = int((di-1)/2)
#         dsj = int((dj-1)/2)
#         # if ds != 0:
#         #     ds = 0
#         # print("dsi", dsi, "dsj", dsj)
#         decal[- dsi, - dsj] = np.sqrt(shape_target[0] * shape_target[1])
#         decalf = dft2(decal)[np.newaxis, ...]
        
#         # psfs_freq = ir2fr(
#         #     psfs_monoch * L_pce[:, np.newaxis, np.newaxis],
#         #     shape=shape_target,
#         #     real=False,
#         # ) * kernel_for_sum_freq * decalf
        
#         psfs_freq = ir2fr(
#             psfs_monoch * L_pce[:, np.newaxis, np.newaxis],
#             shape=shape_target,
#             real=True,
#         ) 

#         self.decalf = decalf
#         self.kernel_for_sum_freq = kernel_for_sum_freq

#         self.di = di
#         self.dj = dj
#         self.shape_target = shape_target
#         self.L_pce = L_pce
#         self.psfs_freq = psfs_freq

#         self.L_specs = L_specs

#         ishape = (L_specs.shape[0], shape_target[0], shape_target[1])
#         oshape = (len(L_pce), shape_target[0]//di, shape_target[1]//dj)

#         print(ishape, oshape)
#         super().__init__(
#             ishape=ishape,
#             oshape=oshape,
#         )


#     def forward(self, maps): # input and output in real, costs 1
#         x = self.maps_to_cube(maps)
#         x_freq = rdft2(x)  # (5, 1, 250, 251)
#         SWCx_freq = self.psfs_freq * (self.kernel_for_sum_freq * self.decalf)[:, :, :((self.shape_target[1] // 2) + 1)] * x_freq
#         SWCx = irdftn(SWCx_freq, self.shape_target)  # (300, 250, 500)

#         SSWCx = SWCx[
#             :, :: self.di, :: self.dj
#         ]  # (300, 50, 100)

#         return SSWCx


#     def adjoint(self, point):
        
#         local_cube = np.zeros(self.ishape)
#         local_cube[:, :: self.di, :: self.dj] = point


#         WCx_freq = rdft2(local_cube)
#         Cx_freq = self.psfs_freq.conj() * (self.kernel_for_sum_freq.conj() * self.decalf.conj())[:, :, :((self.shape_target[1] // 2) + 1)] * WCx_freq

#         x = irdftn(Cx_freq, self.shape_target)

#         maps = self.cube2maps(x)
#         return maps
    
#     def maps_to_cube(self, maps):  # L_specs.shape = (K, N) = (5, 300)
#         cube = np.sum(self.L_specs[:, :, np.newaxis, np.newaxis] * maps[:, np.newaxis, :, :], axis=0)
#         return cube

#     def cube2maps(self, cube):
#         maps = np.concatenate(
#                 [
#                     np.sum(cube * tpl[..., np.newaxis, np.newaxis], axis=0)[np.newaxis, ...]
#                     for tpl in self.L_specs
#                 ],
#                 axis=0,
#                 )
#         return maps


def partitioning_einops2(cube, di, dj):
    new_cube = rearrange(
        cube, "wl (dx bx) (dy by) -> wl (dx dy) bx by", dx=di, dy=dj
    )
    return new_cube

def make_H_spec_freq_sum2(array_psfs, L_pce, L_lamb, L_spec, shape_target, di, dj, pixel_arcsec = 0.111):
    # print("Use of function {}.".format("make_H_spec_freq_sum"))
    weighted_psfs = array_psfs * L_pce[..., np.newaxis, np.newaxis] # (300, 250, 500)
    newaxis_weighted_psfs = weighted_psfs[np.newaxis, ...] # (1, 300, 250, 500)
    
    L_spec = L_spec
    specs = L_spec[..., np.newaxis, np.newaxis] # (5, 300, 1, 1)

    H_spec = newaxis_weighted_psfs * specs # (5, 300, 250, 500)

    H_spec_freq = ir2fr(H_spec, shape_target)
    
    # différence par rapport à make_H_spec_freq est ici
    kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
    # print("avant ir2fr de make_H_spec_freq")
    kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target)[np.newaxis, np.newaxis, ...] # (1, 1, 250, 251)
    # print("après ir2fr de make_H_spec_freq")
    
    return H_spec_freq * kernel_for_sum_freq # (5, 300, 250, 251)

def make_H_spec_freq_sum_full(array_psfs, L_pce, L_lamb, L_spec, shape_target, di, dj, pixel_arcsec = 0.111):
    # print("Use of function {}.".format("make_H_spec_freq_sum"))
    weighted_psfs = array_psfs * L_pce[..., np.newaxis, np.newaxis] # (300, 250, 500)
    newaxis_weighted_psfs = weighted_psfs[np.newaxis, ...] # (1, 300, 250, 500)
    
    L_spec = L_spec
    specs = L_spec[..., np.newaxis, np.newaxis] # (5, 300, 1, 1)

    H_spec = newaxis_weighted_psfs * specs # (5, 300, 250, 500)

    H_spec_freq = ir2fr(H_spec, shape_target, real=False)
    
    # différence par rapport à make_H_spec_freq est ici
    kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
    kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target, real=False)[np.newaxis, np.newaxis, ...] # (1, 1, 250, 251)
    
    return H_spec_freq * kernel_for_sum_freq # (5, 300, 250, 251)


# compute the equivalent of a dot product between a matrix with a shape equal
# to the hessian of the spectro (the hessian itself, its inverse, the hessian for fusion or its inverse) and an input with
# a shape equal to abundance maps (abundance maps themselves or adjoint of data, or sum of adjoint data)
# apply_hessian2, diff with apply_hessian: di and dj different, instead of di == dj
def apply_hessian2(hess_spec_freq, di, dj, shape_target, x, x_is_freq_and_part=False):
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
    concat_HtH_x_freq = concatenating2(HtH_x_freq_sum, shape_target, di, dj)

    # sortie de Fourier
    HtH_x = np.real(idft2(concat_HtH_x_freq))
    # print("HtH_x", HtH_x.shape)

    return HtH_x

# diff with concatenating: now works with decim different for both dimensions
def concatenating2(cubef, shape_target, di, dj):
    n_maps, d1_times_d2, h_block, w_block = cubef.shape
    h, w = shape_target
    
    concatenated_cube = np.zeros((n_maps, h, w), dtype=complex)
    k = 0
    for i in range(di):
        for j in range(dj):
            concatenated_cube[:, i * h_block : (i+1) * h_block, j * w_block : (j+1) * w_block] += cubef[:, k, :, :]
            k += 1
    
    return concatenated_cube
