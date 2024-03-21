import numpy as np
import time
import os

from qmm import QuadObjective, Objective, lcg, mmmg, Huber, HebertLeahy
from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir
import aljabr

from scipy.signal import convolve2d as conv2



class NpDiff_r(aljabr.LinOp): # dim = 3
    def __init__(self, maps_shape):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
    
    def forward(self, x):
        return - np.diff(np.pad(x, ((0, 0), (1, 0), (0, 0)), 'wrap'), axis=1)
    
    def adjoint(self, y):
        return np.diff(np.pad(y, ((0, 0), (0, 1), (0, 0)), 'wrap'), axis=1)

class NpDiff_c(aljabr.LinOp): # dim = 3
    def __init__(self, maps_shape):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
    
    def forward(self, x):
        return - np.diff(np.pad(x, ((0, 0), (0, 0), (1, 0)), 'wrap'), axis=2)
    
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
        model_spectro,
        mu_reg,
        printing=False,
        gradient="separated",
    ):
        self.mu_spectro = mu_spectro
        self.y_spectro = y_spectro
        self.model_spectro = model_spectro

        n_spec = model_spectro.tpls.shape[0]
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

        shape_target = model_spectro.ishape[1:]
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


    def run_lcg(self, maximum_iterations, tolerance=1e-12, calc_crit = False, perf_crit = None, value_init = 0.5):
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

        self.L_crit_val_lcg = []
        self.L_perf_crit = []
        
        def perf_crit_for_lcg(res_lcg):
            x_hat = res_lcg.x.reshape(self.shape_of_output)
            self.L_perf_crit.append(perf_crit(x_hat))

        def print_last_grad_norm(res_lcg):
            print(res_lcg.grad_norm[-1])

        list_obj = [spectro_data_adeq] + prior
        
        t1 = time.time()

        if calc_crit and perf_crit == None:
            print("LCG : Criterion calculated at each iteration!")
            # res_lcg = lcg(
            #     imager_data_adeq + spectro_data_adeq + prior,
            #     init,
            #     tol=tolerance,
            #     max_iter=maximum_iterations,
            #     calc_objv = True
            # )
            res_lcg = lcg(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = self.crit_val_for_lcg
            )
        elif calc_crit == False and perf_crit != None:
            print("LCG : perf_crit calculated at each iteration!")
            res_lcg = lcg(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = print_last_grad_norm
            )
        elif calc_crit and perf_crit != None:
            print("Criterion to calculate AND performance criterion to calculate ?")
            return None
        elif calc_crit == False and perf_crit == None:
            res_lcg = lcg(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations
            )
        if self.printing:
            print("Total time needed for LCG :", round(time.time() - t1, 3))
        
        # running_time = time.time() - t1

        return res_lcg  # , running_time
    

    def run_mmmg(self, maximum_iterations, tolerance=1e-12, calc_crit = False, perf_crit = None, value_init = 0.5):
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

        self.L_crit_val_lcg = []
        self.L_perf_crit = []

        def print_last_grad_norm(res_lcg):
            print(res_lcg.grad_norm[-1])

        list_obj = [spectro_data_adeq] + prior
        
        t1 = time.time()

        if calc_crit and perf_crit == None:
            print("LCG : Criterion calculated at each iteration!")
            # res_lcg = lcg(
            #     imager_data_adeq + spectro_data_adeq + prior,
            #     init,
            #     tol=tolerance,
            #     max_iter=maximum_iterations,
            #     calc_objv = True
            # )
            res_mmmg = mmmg(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = self.crit_val
            )
        elif calc_crit == False and perf_crit != None:
            print("LCG : perf_crit calculated at each iteration!")
            res_mmmg = mmmg(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = print_last_grad_norm
            )
        elif calc_crit and perf_crit != None:
            print("Criterion to calculate AND performance criterion to calculate ?")
            return None
        elif calc_crit == False and perf_crit == None:
            res_mmmg = mmmg(
                list_obj,
                init,
                tol=tolerance,
                max_iter=maximum_iterations
            )
        if self.printing:
            print("Total time needed for MMMG :", round(time.time() - t1, 3))
        
        # running_time = time.time() - t1

        return res_mmmg  # , running_time
    
    def crit_val(self, x_hat):
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
    
    def crit_val_for_lcg(self, res_lcg):
        x_hat = res_lcg.x.reshape(self.shape_of_output)
        self.L_crit_val_lcg.append(self.crit_val(x_hat))

    
