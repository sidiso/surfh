import numpy as np
import time
import os

from qmm import QuadObjective, Objective, lcg, mmmg, Huber, HebertLeahy
from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir
from udft import idft2, dft2, ir2fr, rdft2, irdftn, laplacian
import aljabr
from surfh.ToolsDir import utils
from surfh.ToolsDir import algorithms

from scipy.signal import convolve2d as conv2

import matplotlib.pyplot as plt

import inspect

class NpDiff_r(aljabr.LinOp): # dim = 3
    def __init__(self, maps_shape, mask=None):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
        self.mask = mask
    
    def forward(self, x):
        Dx_masked = - np.diff(np.pad(x, ((0, 0), (1, 0), (0, 0)), 'wrap'), axis=1)
        return Dx_masked

    def adjoint(self, y):    
        Dy_masked = np.diff(np.pad(y, ((0, 0), (0, 1), (0, 0)), 'wrap'), axis=1)
        return Dy_masked

class NpDiff_c(aljabr.LinOp): # dim = 3
    def __init__(self, maps_shape, mask=None):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
        self.mask=mask

    def forward(self, x):
        Dx_masked = - np.diff(np.pad(x, ((0, 0), (0, 0), (1, 0)), 'wrap'), axis=2)        
        return Dx_masked   

    def adjoint(self, y):
        return np.diff(np.pad(y, ((0, 0), (0, 0), (0, 1)), 'wrap'), axis=2)

class Difference_Operator_Joint:  # gradients are joint
    def __init__(self, shape_target):
        diff_kernel = laplacian(2)
        D_freq = ir2fr(diff_kernel, shape=shape_target, real=True)
        self.D_freq = D_freq

    def D(self, x):
        return irdftn(self.D_freq[np.newaxis, ...] * rdft2(x), shape=x.shape[1:])

    def D_t(self, x):
        return irdftn(
            np.conj(self.D_freq[np.newaxis, ...]) * rdft2(x), shape=x.shape[1:]
        )

    def DtD(self, x):
        return irdftn(
            np.abs(self.D_freq[np.newaxis, ...]) ** 2 * rdft2(x), shape=x.shape[1:]
        )


#%%

class QuadCriterion_MRS:
    # y_imager must be compatible with fusion model, i.e. use mirim_model_for_fusion here
    def __init__(
        self,
        mu_spectro,
        y_spectro,
        model_mixing,
        mu_reg,
        mask,
        printing=False,
        gradient="separated",
    ):
        self.mu_spectro = mu_spectro
        self.y_spectro = y_spectro
        self.model_spectro = model_mixing

        n_spec = model_mixing.ishape[0]
        self.n_spec = n_spec
        self.it = 1
        self.mask = mask

        assert (
            type(mu_reg) == float
            or type(mu_reg) == int
            or type(mu_reg) == list
            or type(mu_reg) == np.ndarray
        )
        self.mu_reg = mu_reg
        if type(mu_reg) == list or type(mu_reg) == np.ndarray:
            assert len(mu_reg) == n_spec

        shape_target = model_mixing.ishape[1:]
        shape_of_output = (n_spec, shape_target[0], shape_target[1])
        self.shape_of_output = shape_of_output

        if gradient == "joint":
            diff_op_joint = Difference_Operator_Joint(shape_target)
            self.diff_op_joint = diff_op_joint
        elif gradient == "separated":
            npdiff_r = NpDiff_r(shape_of_output, self.mask)
            self.npdiff_r = npdiff_r
            npdiff_c = NpDiff_c(shape_of_output, self.mask)
            self.npdiff_c = npdiff_c

        if type(self.mu_reg) == list or type(self.mu_reg) == np.ndarray:
            L_mu = np.copy(self.mu_reg)
        elif type(self.mu_reg) == int or type(self.mu_reg) == float:
            L_mu = np.ones(self.n_spec) * self.mu_reg  # same mu for all maps
        self.L_mu = L_mu

        self.printing = printing
        self.gradient = gradient

    def run_method(self, method='lcg', maximum_iterations=10, tolerance=1e-12, calc_crit = False, perf_crit = None, value_init = 0.5):
        assert type(self.mu_reg) == int or type(self.mu_reg) == float
        # lcg codé que avec un hyper paramètre

        if type(value_init) == float or type(value_init) == int:
            init = np.ones(self.shape_of_output) * value_init
        elif type(value_init) == np.ndarray:
            assert value_init.shape == self.shape_of_output
            init = value_init

        # t1 = time.time()

        spectro_data_adeq = QuadObjective(
            self.model_spectro.forward,
            self.model_spectro.adjoint,
            data=self.y_spectro,
            hyper=self.mu_spectro,
            name="Spectro",
        )
        if self.gradient == "joint":  # regularization term with joint gradients
            prior = QuadObjective(
                self.diff_op_joint.D,
                self.diff_op_joint.D_t,
                self.diff_op_joint.DtD,
                hyper=self.mu_reg,
                name="Reg joint",
            )
            prior = [prior]

        elif self.gradient == "separated":
            prior_r = QuadObjective(
                self.npdiff_r.forward,
                self.npdiff_r.adjoint,
                hyper=self.mu_reg,
            )
            prior_c = QuadObjective(
                self.npdiff_c.forward,
                self.npdiff_c.adjoint,
                hyper=self.mu_reg,
            )
            prior = [prior_r, prior_c]

        self.L_crit_val = []
        
        def perf_crit_with_reshape(res):
            x_hat = res.x.reshape(self.shape_of_output)
            crit_val = self.get_crit_val(x_hat)
            self.L_crit_val.append(crit_val)
            print(f"Criterion value = {crit_val}\n")

        def print_last_grad_norm(res):
            print(f"Iteration n°{self.it}, Grad norm = {res.grad_norm[-1]}")
            self.it = self.it + 1

        def print_last_grad_norm_and_crit(res):
            print_last_grad_norm(res)
            if self.it%5 == 2:
                perf_crit_with_reshape(res)


        list_obj = [spectro_data_adeq] + prior
        t1 = time.time()

        if method == 'lcg':
            function = lcg    
        else:
            function = mmmg

        if calc_crit and perf_crit == None:
            print(f"{method} : Criterion calculated at each iteration!")
            res = function(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = self.get_crit_val
            )
        elif calc_crit == False and perf_crit != None:
            print(f"{method} : perf_crit calculated at each iteration!")
            res = function(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = print_last_grad_norm
            )
        elif calc_crit and perf_crit != None:
            print(f"{method} : criterion and gradient printed at each iteration!")
            res = function(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = print_last_grad_norm_and_crit
            )
        elif calc_crit == False and perf_crit == None:
            res = function(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations
            )
        if self.printing:
            print(f"Total time needed for {method} :", round(time.time() - t1, 3))
        
        # running_time = time.time() - t1

        return res  # , running_time


    
    def get_crit_val(self, x_hat):
        # data_term_imager = self.mu_imager * np.sum(
        #     (self.y_imager - self.model_imager.forward(x_hat)) ** 2
        # )
        data_term_spectro = self.mu_spectro * np.sum(
            (self.y_spectro - self.model_spectro.forward(x_hat)) ** 2
        )

        if self.gradient == "joint":
            regul_term = self.mu_reg * np.sum((self.diff_op_joint.D(x_hat)) ** 2)
            
        elif self.gradient == "separated":
            regul_term = self.mu_reg * (
                np.sum(
                    self.npdiff_r.forward(x_hat) ** 2
                    + self.npdiff_c.forward(x_hat) ** 2
                )
            )

        # J_val = (data_term_imager + data_term_spectro + regul_term) / 2
        J_val = (data_term_spectro + regul_term) / 2
        # on divise par 2 par convention, afin de ne pas trouver un 1/2 dans le calcul de dérivée

        return J_val

    


# intérêt par rapport à QuadCriterion: modèles mis à jour pour décimation libre
class QuadCriterion3:
    # y_imager must be compatible with fusion model, i.e. use mirim_model_for_fusion here
    def __init__(
        self,
        data,
        model,
        mu_reg,
        printing=False,
        gradient="separated",
    ):
        self.data = data
        self.model = model

        n_spec = model.n_spec
        self.n_spec = n_spec

        assert (
            type(mu_reg) == float
            or type(mu_reg) == int
            or type(mu_reg) == list
            or type(mu_reg) == np.ndarray
        )
        self.mu_reg = mu_reg
        if type(mu_reg) == list or type(mu_reg) == np.ndarray:
            assert len(mu_reg) == n_spec

        shape_target = model.shape_target
        shape_of_output = (n_spec, shape_target[0], shape_target[1])
        self.shape_of_output = shape_of_output

        if gradient == "joint":
            diff_op_joint = Difference_Operator_Joint(shape_target)
            self.diff_op_joint = diff_op_joint
        elif gradient == "separated":
            npdiff_r = NpDiff_r(shape_of_output)
            self.npdiff_r = npdiff_r
            npdiff_c = NpDiff_c(shape_of_output)
            self.npdiff_c = npdiff_c

        if type(self.mu_reg) == list or type(self.mu_reg) == np.ndarray:
            L_mu = np.copy(self.mu_reg)
        elif type(self.mu_reg) == int or type(self.mu_reg) == float:
            L_mu = np.ones(self.n_spec) * self.mu_reg  # same mu for all maps
        self.L_mu = L_mu

        self.printing = printing
        self.gradient = gradient

    def run_expsol(self):
        if self.printing:
            # print("Preprocessing starts...")
            t1 = time.time()
        
        if type(self.mu_reg) == list or type(self.mu_reg) == np.ndarray:
            L_mu = np.copy(self.mu_reg)
        elif type(self.mu_reg) == int or type(self.mu_reg) == float:
            L_mu = np.ones(self.n_spec) * self.mu_reg  # same mu for all maps
        
        regul_fusion_model = Regul_Fusion_Model3(
            self.model,
            L_mu,
            gradient=self.gradient,
        )

        inv_fusion_model = Inv_Regul_Fusion_Model3(regul_fusion_model)

        if self.printing:
            t2 = time.time()
            time_preprocess = round(t2 - t1, 3)
            # print("Preprocessing ended in {} sec.".format(time_preprocess))

        res_with_all_data = inv_fusion_model.map_reconstruction(
            self.data
        )

        if self.printing:
            t3 = time.time()
            time_all = round(t3 - t1, 3)
            time_calc = round(time_all - time_preprocess, 3)
            print(
                "Total time needed for expsol = {} + {} = {} sec.".format(
                    time_preprocess, time_calc, time_all
                )
            )

        return res_with_all_data
    


# separation of classes for Regul and Inversion because Regul without inversion will be useful for iterative methods        
class Regul_Fusion_Model3():
    def __init__(self, model, L_mu_reg, gradient="joint"):
        # print("Use of class {}.".format("Regul_Fusion_Model"))
        # making hessian for fusion
        hessian_fusion = model.hess_spec_freq
        shape_target = model.shape_target
        
        # adding regularization
        if gradient == "joint":
            diff_kernel = laplacian(2)
            D_freq = ir2fr(diff_kernel, shape=shape_target, real=False)
            part_D_freq = algorithms.partitioning_einops2(D_freq[np.newaxis, ...], model.di, model.dj)[0]
        
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
            part_D_freq_row = algorithms.partitioning_einops2(D_freq_row[np.newaxis, ...], model.di, model.dj)[0]
            part_D_freq_col = algorithms.partitioning_einops2(D_freq_col[np.newaxis, ...], model.di, model.dj)[0]
            
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
        self.di = model.di
        self.dj = model.dj
        self.shape_target = shape_target
        
        self.model = model
        

class Inv_Regul_Fusion_Model3():
    def __init__(self, regul_fusion_model):
        # print("Use of class {}.".format("Inv_Regul_Fusion_Model"))
        # make inverse of hessian        
        inv_hess_fusion = algorithms.make_iHtH_spectro(regul_fusion_model.regul_hess_fusion)
        self.inv_hess_fusion = inv_hess_fusion
        
        # real_inv_hessian = idft2(inv_hess_fusion)
        # inv_hess_fusion_half = rdft2(real_inv_hessian)
        # self.inv_hess_fusion_half = inv_hess_fusion_half
        
        self.model = regul_fusion_model.model
        
        self.di = regul_fusion_model.di
        self.dj = regul_fusion_model.dj
        
        self.shape_target = regul_fusion_model.shape_target
        
    def map_reconstruction(self, data):
        # mirim_adjoint_data = self.mirim_model_for_fusion.adjoint_freq_full(mirim_obs_data)
        # spectro_adjoint_data = self.spectro_model.adjoint_freq_full(spectro_obs_data)
        # adjoint_data_freq = self.mu_mirim * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data
        
        # t1 = time.time()
        
        model_adjoint_data = self.model.adjoint(data)
        adjoint_data_freq = dft2(model_adjoint_data)
        
        # t2 = time.time()
        
        res = np.real(idft2(algorithms.apply_hessian_freq(self.inv_hess_fusion, self.di, self.dj, self.shape_target, adjoint_data_freq)))
        
        # t3 = time.time()
        
        # print("map_reconstruction : temps calcul de b =", t2 - t1)
        # print("map_reconstruction : temps calcul de Q-1 b =", t3 - t2)
        
        return res 
