import numpy as np
import os
import matplotlib.pyplot as plt
from aljabr import LinOp

from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir, dft2, dftn, idft2
from einops import einsum, rearrange, reduce, repeat

from surfh.Signalprocessing.utilities import partitioning_einops2

import time


class Spectro_Model_3(LinOp):
    def __init__(self, 
                    psfs_monoch, 
                    L_pce, 
                    di, dj, 
                    lamb_cube, 
                    L_specs, 
                    shape_target, 
                    pixel_arcsec=0.111 # size of pixels before integration and decimation of spectro
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
        L_specs_converted = self.unit_conversion(L_specs)
        specs = L_specs_converted[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]  # (5, 300, 1, 1)
        # print("check1")
        n_spec = specs.shape[0]
        HtH_freq = np.zeros(
            (n_spec, n_spec, di * dj, di * dj, h_block, w_block), dtype=complex
        )
        # # print("check2")
        # print(f"Shape for things needed for HtH_freq: specs {specs.shape}, mat {mat.shape}, HtH_freq {HtH_freq.shape}, n_spec {n_spec}")
        # start = time.time()
        # for k1 in range(n_spec):
        #     for k2 in range(k1, n_spec):
        #         HtH_freq[k1, k2] += np.sum(specs[k1] * specs[k2] * mat, axis=0)
        # end = time.time()
        # print(f"Time to create HtH_freq: {end - start} seconds")

        # # print("check3")
        # # utilisation de la symétrie de HtH
        # start = time.time()
        # for k1 in range(n_spec):
        #     for k2 in range(k1):
        #         HtH_freq[k1, k2] += HtH_freq[k2, k1]
        # end = time.time()
        # print(f"Time to symmetrize HtH_freq: {end - start} seconds")



        HtH_freq2 = np.zeros(
            (n_spec, n_spec, di * dj, di * dj, h_block, w_block), dtype=complex
        )
        start = time.time()
        specs2 = specs[:, :, 0, 0, 0, 0]   # (13, 995)

        HtH_freq = np.einsum(
            'ki,li,iabcd->klabcd',
            specs2,
            specs2,
            mat,
            optimize=True
        )

        end = time.time()
        print(f"Time to create HtH_freq with einsum: {end - start} seconds")

        self.hess_spec_freq = HtH_freq

        # H_spec_freq utile pour forward et adjoint
        # self.H_spec_freq = make_H_spec_freq_sum2(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # )
        
        # print("2 x H_spec_freq enlevés !!")
        
        start = time.time()
        print(f'Shape for things needed for H_spec_freq: psfs_monoch {psfs_monoch.shape}, L_pce {L_pce.shape}, lamb_cube {lamb_cube.shape}, L_specs {L_specs.shape}, shape_target {shape_target}, di {di}, dj {dj}')
        self.H_spec_freq = self.make_H_spec_freq_sum2_safe(
            psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        ) 
        print("Step 4")
        kernel = rdft2(decal)[np.newaxis, np.newaxis, :, :]
        self.H_spec_freq *= kernel

        # self.H_spec_freq = self.H_spec_freq* rdft2(decal)[np.newaxis, np.newaxis, :, :]
        print("Step 5")

        # self.H_spec_freq = self.make_H_spec_freq_sum2(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # ) * rdft2(decal)[np.newaxis, np.newaxis, :, :]
        # end = time.time()
        print(f"Time to create H_spec_freq: {end - start} seconds")
        # print(f"Are H_spec_freq and H_spec_freq2 close? {np.allclose(self.H_spec_freq, self.H_spec_freq2)}")
        # diff = np.max(np.abs(self.H_spec_freq - self.H_spec_freq2))
        # ref  = np.max(np.abs(self.H_spec_freq))

        # print("max abs diff :", diff)
        # print("relative err :", diff / ref)

        # # utile pour forward_freq_to_freq et forward_freq_to_real
        # start = time.time()
        # self.H_spec_freq_full = self.make_H_spec_freq_sum_full(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # ) * decalf[np.newaxis, np.newaxis, :, :]
        # end = time.time()
        # print(f"Time to create H_spec_freq and H_spec_freq_full: {end - start} seconds")

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
        # H_spec_x_freq = einsum(
        #     np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        H_spec_x_freq = np.einsum(
            "tlij,blij->tij",
            self.H_spec_freq.conj(),
            original_cube_freq,
            optimize=True
        )

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
    

    def fwadj(self, x):
        assert x.shape == self.ishape
        return self.apply_hessian2(self.hess_spec_freq, self.di, self.dj, self.shape_target, x)


    def unit_conversion(self, flux):
        return flux

    def make_H_spec_freq_sum2(self, array_psfs, L_pce, L_lamb, L_spec, shape_target, di, dj, pixel_arcsec = 0.111):
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



    def make_H_spec_freq_sum_full(self, array_psfs, L_pce, L_lamb, L_spec, shape_target, di, dj, pixel_arcsec = 0.111):
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




    def make_H_spec_freq_sum2_safe(
        self,
        array_psfs,
        L_pce,
        L_lamb,
        L_spec,
        shape_target,
        di,
        dj,
        chunk_size=256
    ):

        n_spec = L_spec.shape[0]
        n_lambda = array_psfs.shape[0]

        H, W = shape_target
        Wf = W//2 + 1
        print('Step 1')
        H_spec_freq = np.empty(
            (n_spec, n_lambda, H, Wf),
            dtype=np.complex64
        )
        print('Step 2')
        # kernel FFT UNE SEULE FOIS
        kernel = np.ones((di, dj), dtype=np.float32)
        kernel_freq = ir2fr(kernel, shape_target)[None, None]

        for start in range(0, n_lambda, chunk_size):
            print(f'Processing chunk from {start} to {min(start + chunk_size, n_lambda)}')
            end = min(start + chunk_size, n_lambda)

            # ----- PSF -----
            psf_chunk = array_psfs[start:end].astype(np.float32)

            psf_freq = ir2fr(psf_chunk, shape_target).astype(np.complex64)

            # pondération PCE
            weights = L_spec[:, start:end] * L_pce[start:end]

            H_spec_freq[:, start:end] = np.einsum(
                "lhw,tl->tlhw",
                psf_freq,
                weights,
                optimize=True
            )* kernel_freq
        print('Step 3')
        # H_spec_freq *= kernel_freq
        return H_spec_freq 
