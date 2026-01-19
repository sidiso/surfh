import time
import numpy as np


from surfh.Models.simplifiedMIRIM import Mirim_Model_For_Fusion
from surfh.Models.simplifiedMRS import Spectro_Model_3
from surfh.Algorithm.diff_operators import NpDiff_c, NpDiff_r
from surfh.Algorithm.fusion_tools import Regul_Fusion_Model2
from surfh.Algorithm.fusion_tools import Inv_Regul_Fusion_Model2


class QuadCriterion2:
    # y_imager must be compatible with fusion model, i.e. use mirim_model_for_fusion here
    def __init__(
        self,
        mu_imager,
        y_imager,
        model_imager: Mirim_Model_For_Fusion,
        mu_spectro,
        y_spectro,
        model_spectro: Spectro_Model_3,
        mu_reg,
        printing=False,
        gradient="separated",
    ):
        self.mu_imager = mu_imager
        self.y_imager = y_imager
        self.model_imager = model_imager
        self.mu_spectro = mu_spectro
        self.y_spectro = y_spectro
        self.model_spectro = model_spectro

        n_spec = model_imager.part_hess_mirim_freq_full.shape[0]
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

        shape_target = model_imager.shape_target
        shape_of_output = (n_spec, shape_target[0], shape_target[1])
        self.shape_of_output = shape_of_output

        if gradient == "joint":
            raise RuntimeError("Choose Separated gradient, Joint is not implemented.")
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
        
        regul_fusion_model = Regul_Fusion_Model2(
            self.model_imager,
            self.model_spectro,
            L_mu,
            self.mu_imager,
            self.mu_spectro,
            gradient=self.gradient,
        )

        inv_fusion_model = Inv_Regul_Fusion_Model2(regul_fusion_model)

        if self.printing:
            t2 = time.time()
            time_preprocess = round(t2 - t1, 3)
            # print("Preprocessing ended in {} sec.".format(time_preprocess))

        res_with_all_data = inv_fusion_model.map_reconstruction(
            self.y_imager, self.y_spectro
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

    def crit_val(self, x_hat):
        data_term_imager = self.mu_imager * np.sum(
            (self.y_imager - self.model_imager.forward(x_hat)) ** 2
        )
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

        J_val = (data_term_imager + data_term_spectro + regul_term) / 2
        # on divise par 2 par convention, afin de ne pas trouver un 1/2 dans le calcul de dérivée

        return J_val
    
    def crit_val_for_lcg(self, res_lcg):
        x_hat = res_lcg.x.reshape(self.shape_of_output)
        self.L_crit_val_lcg.append(self.crit_val(x_hat))