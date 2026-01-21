import numpy as np
import time
from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir, dftn, idft2, dft2

from surfh.Models.simplifiedMIRIM import Mirim_Model_For_Fusion
from surfh.Models.simplifiedMRS import Spectro_Model_3
from surfh.Signalprocessing.utilities import partitioning_einops2, make_iHtH_spectro, apply_hessian_freq, make_iHtH_spectro_fast, make_iHtH_spectro_gpu_torch


# separation of classes for Regul and Inversion because Regul without inversion will be useful for iterative methods        
class Regul_Fusion_Model2():
    def __init__(self, mirim_model_for_fusion:Mirim_Model_For_Fusion, spectro_model:Spectro_Model_3, L_mu_reg, mu_mirim, mu_spectro, gradient="joint"):
        # print("Use of class {}.".format("Regul_Fusion_Model"))
        # making hessian for fusion
        hessian_fusion = mu_mirim * mirim_model_for_fusion.part_hess_mirim_freq_full + mu_spectro * spectro_model.hess_spec_freq
        
        # adding regularization
        shape_target = spectro_model.shape_target
        if gradient == "joint":
            diff_kernel = laplacian(2)
            D_freq = ir2fr(diff_kernel, shape=shape_target, real=False)
            part_D_freq = partitioning_einops2(D_freq[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
        
            regul_hess_fusion = np.copy(hessian_fusion)
            n_spec, _, di_times_dj, _, h_block, w_block = regul_hess_fusion.shape
            for k in range(n_spec):
                for i in range(di_times_dj):
                    regul_hess_fusion[k, k, i, i, :, :] += (
                        L_mu_reg[k] * (np.abs(part_D_freq[i]) ** 2)
                    )
                    
        elif gradient == "separated":
            diff_kernel_row = (np.array([-1, 1]))[..., np.newaxis]
            diff_kernel_col = (np.array([-1, 1]))[np.newaxis, ...]

            D_freq_row = ir2fr(diff_kernel_row, shape=shape_target, real=False)
            D_freq_col = ir2fr(diff_kernel_col, shape=shape_target, real=False)
            
            # partitioning
            part_D_freq_row = partitioning_einops2(D_freq_row[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
            part_D_freq_col = partitioning_einops2(D_freq_col[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
            
            # summing on diagonal of hessian
            regul_hess_fusion = np.copy(hessian_fusion)
            n_spec, _, di_times_dj, _, h_block, w_block = regul_hess_fusion.shape
            for k in range(n_spec):
                for i in range(di_times_dj):
                    coeff = L_mu_reg[k]
                    regul_hess_fusion[k, k, i, i, :, :] += (
                        coeff * (np.abs(part_D_freq_row[i])**2 + np.abs(part_D_freq_col[i])**2)
                    )
                
        self.regul_hess_fusion = regul_hess_fusion
        self.di = spectro_model.di
        self.dj = spectro_model.dj
        self.shape_target = shape_target
        
        self.mirim_model_for_fusion = mirim_model_for_fusion
        self.mu_mirim = mu_mirim
        
        self.spectro_model = spectro_model
        self.mu_spectro = mu_spectro




class Inv_Regul_Fusion_Model2():
    def __init__(self, regul_fusion_model:Regul_Fusion_Model2):
        # print("Use of class {}.".format("Inv_Regul_Fusion_Model"))
        # make inverse of hessian        
        # s = time.time()
        # inv_hess_fusion = make_iHtH_spectro(regul_fusion_model.regul_hess_fusion)
        # e = time.time()
        # print("Inversion of HtH hessian time:", e - s, "s")

        s = time.time()
        inv_hess_fusion = make_iHtH_spectro_fast(regul_fusion_model.regul_hess_fusion)
        e = time.time()
        print("Inversion of HtH hessian Fast time:", e - s, "s")        

        # import torch
        # s = time.time()
        # inv_hess_fusion2 = make_iHtH_spectro_gpu_torch(regul_fusion_model.regul_hess_fusion)  # retourne torch cuda

        # # inv_hess_fusion2 = make_iHtH_spectro_gpu_torch(regul_fusion_model.regul_hess_fusion)
        # e = time.time()
        # print("Inversion of HtH hessian GPU time:", e - s, "s")
        # print("Both methods give the same result: ", np.allclose(inv_hess_fusion, inv_hess_fusion2))

        # print(f"Both methods give the same result: {np.allclose(inv_hess_fusion, inv_hess_fusion2)}")
        self.inv_hess_fusion = inv_hess_fusion
        
        # real_inv_hessian = idft2(inv_hess_fusion)
        # inv_hess_fusion_half = rdft2(real_inv_hessian)
        # self.inv_hess_fusion_half = inv_hess_fusion_half
        
        self.mirim_model_for_fusion = regul_fusion_model.mirim_model_for_fusion
        self.mu_mirim = regul_fusion_model.mu_mirim
        
        self.spectro_model = regul_fusion_model.spectro_model
        self.mu_spectro = regul_fusion_model.mu_spectro
        
        self.di = regul_fusion_model.di
        self.dj = regul_fusion_model.dj
        
        self.shape_target = regul_fusion_model.shape_target
        
    def map_reconstruction(self, mirim_obs_data, spectro_obs_data):
        # mirim_adjoint_data = self.mirim_model_for_fusion.adjoint_freq_full(mirim_obs_data)
        # spectro_adjoint_data = self.spectro_model.adjoint_freq_full(spectro_obs_data)
        # adjoint_data_freq = self.mu_mirim * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data
        
        # t1 = time.time()
        
        mirim_adjoint_data = self.mirim_model_for_fusion.adjoint(mirim_obs_data)
        spectro_adjoint_data = self.spectro_model.adjoint(spectro_obs_data)
        adjoint_data_freq = dft2(self.mu_mirim * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data)
        
        # t2 = time.time()
        
        res = np.real(idft2(apply_hessian_freq(self.inv_hess_fusion, self.di, self.dj, self.shape_target, adjoint_data_freq)))
        
        # t3 = time.time()
        
        # print("map_reconstruction : temps calcul de b =", t2 - t1)
        # print("map_reconstruction : temps calcul de Q-1 b =", t3 - t2)
        
        return res 