import matplotlib.pyplot as plt
import numpy as np
import time
import os
import udft
from pathlib import Path

from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import Angle

import scipy
from scipy.signal import convolve2d as conv2

from surfh import instru, models
from surfh import utils
from surfh import realmiri
from pathlib import Path

from qmm import QuadObjective, Objective, lcg, mmmg, Huber, HebertLeahy
from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir
import aljabr


# Data load and save directories 
fits_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_fits/'
numpy_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_numpy/'
slices_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_numpy_slices/'
psf_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_psf/'
result_directory = ''


# Create result cube
def orion():
    """Rerturn maps, templates, spatial step and wavelength"""
    maps = fits.open("./cube_orion/abundances_orion.fits")[0].data

    h2_map = maps[0]
    if_map = maps[1]
    df_map = maps[2]
    mc_map = maps[3]

    spectrums = fits.open("./cube_orion/spectra_mir_orion.fits")[1].data
    wavel_axis = spectrums.wavelength

    h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
    if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
    df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
    mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

    return (
        np.asarray((h2_map, if_map, df_map, mc_map)),
        np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
        0.025,
        wavel_axis,
    )


maps, tpl, step, wavel_axis = orion()
spatial_subsampling = 4
impulse_response = np.ones((spatial_subsampling, spatial_subsampling)) / spatial_subsampling ** 2
maps = np.asarray([conv2(arr, impulse_response)[::spatial_subsampling, ::spatial_subsampling] for arr in maps])
step_Angle = Angle(step, u.arcsec)


"""
Set Cube coordinate.
"""
margin=100
maps_shape = (maps.shape[0], maps.shape[1]+margin*2, maps.shape[2]+margin*2)
step_Angle = Angle(step, u.arcsec)
origin_alpha_axis = np.arange(maps_shape[1]) * step_Angle.degree
origin_beta_axis = np.arange(maps_shape[2]) * step_Angle.degree
origin_alpha_axis -= np.mean(origin_alpha_axis)
origin_beta_axis -= np.mean(origin_beta_axis)

tpl_ss = 3
wavel_axis = wavel_axis[::tpl_ss]
#spsf = utils.gaussian_psf(wavel_axis, step_Angle.degree)
spsf = np.load(psf_directory + 'psfs_pixscale0.025_fov11.25_date_300123.npy')

if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps_shape[1:])

"""
Process Metadata for all Fits in directory
"""
main_pointing = instru.Coord(0, 0) # Set the main pointing from the instrument FoV
pointings = instru.CoordList([main_pointing])#.pix(step_Angle.degree) # We do not use dithering for first tests
list_channels = []
list_data = []

for file in os.listdir(fits_directory):
    split_file  = file.split('_')

    # Create IFU for specific fits
    list_channels.append(realmiri.get_IFU(fits_directory + '/' + file))

    # Load and set NaN to 0
    data = np.load(slices_directory + Path(file).stem + '.npy')
    data[np.where(np.isnan(data))] = 0

    list_data.append(data)


chans = []
forward_data = []
sorted_chan_data = [a for a in sorted((tup.name, tup, da) for tup, da in zip(list_channels, list_data))]
for i in range(9):
    chans.append(sorted_chan_data[i][1])
    forward_data.append(sorted_chan_data[i][2].ravel())

array_data = np.concatenate(forward_data, axis=0)

spectro = models.SpectroLMM(
    chans, # List of channels and bands 
    origin_alpha_axis, # Alpha Coordinates of the cube
    origin_beta_axis, # Beta Coordinates of the cube
    wavel_axis, # Wavelength axis of the cube
    sotf, # Optical PSF
    pointings, # List of pointing (mainly used for dithering)
    tpl,
    verbose=True,
    serial=False,
)


"""
Fusion 
"""

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

        n_spec = model_spectro.tpls[0]
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

        shape_target = model_spectro
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

        init = np.ones(self.shape_of_output) * value_init

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
            prior = prior_r + prior_c

        self.L_crit_val_lcg = []
        self.L_perf_crit = []
        
        def perf_crit_for_lcg(res_lcg):
            x_hat = res_lcg.x.reshape(self.shape_of_output)
            self.L_perf_crit.append(perf_crit(x_hat))
        
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
                spectro_data_adeq + prior,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = self.crit_val_for_lcg
            )
        elif calc_crit == False and perf_crit != None:
            print("LCG : perf_crit calculated at each iteration!")
            res_lcg = lcg(
                spectro_data_adeq + prior,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = perf_crit_for_lcg
            )
        elif calc_crit and perf_crit != None:
            print("Criterion to calculate AND performance criterion to calculate ?")
            return None
        elif calc_crit == False and perf_crit == None:
            res_lcg = lcg(
                spectro_data_adeq + prior,
                init,
                tol=tolerance,
                max_iter=maximum_iterations
            )
        if self.printing:
            print("Total time needed for LCG :", round(time.time() - t1, 3))
        
        # running_time = time.time() - t1

        return res_lcg  # , running_time
    
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



quadCrit = QuadCriterion_MRS(1, array_data, spectro, 0.1, True)
res_lcg = quadCrit.run_lcg(5)



