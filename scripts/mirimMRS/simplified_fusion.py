import numpy as np
import os
import udft
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pathlib

from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir, dft2, dftn, idft2

from aljabr import dottest
from surfh.Models import simplifiedMRS, simplifiedMIRIM
from surfh.Algorithm.criterion_spectroImageur import QuadCriterion_spectroImageur
from surfh.ToolsDir import utils

def maps_to_cube(maps, L_specs):  # L_specs.shape = (K, N) = (5, 300)
    cube = np.sum(L_specs[:, :, np.newaxis, np.newaxis] * maps[:, np.newaxis, :, :], axis=0)
    return cube

def reconstruction_method(imageurModel, mirim_data, spectroModel, mrs_data, result_path, hyperParameter, niter, method, bool_templates):
    """
    Perform the reconstruction method and save results.

    Parameters:
        spectroModel: Spectro model object
        ndata: Data array
        templates: Templates array
        pointings: Pointings data
        result_path: Path to save results
        wavel_axis: Wavelength axis array
    """
    # Hyperparameters
    # hyperParameter = 5e3
    # method = "lcg"
    # niter = 50
    value_init = 0
    niter = 100
    hyperParameter = 0.00001
    # # Create result directory
    # result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_4_lmm_{True}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}/'
    # path = pathlib.Path(result_path + result_dir)
    # path.mkdir(parents=True, exist_ok=True)

    quadCrit_fusion = QuadCriterion_spectroImageur(
        mu_imager=1.0,
        y_imager=mirim_data,
        model_imager=imageurModel,
        mu_spectro=1.0,
        y_spectro=mrs_data,
        model_spectro=spectroModel,
        mu_reg=hyperParameter,
        printing=True,
        gradient='separated'
    )

    res_fusion = quadCrit_fusion.run_lcg(maximum_iterations=niter, 
                                         perf_crit=None, 
                                         calc_crit=True, 
                                         value_init=value_init)

    # print(f"Results save in {path}")
    # # Save results
    # if bool_templates is False:
    #     print("No templates, save only cube")
    #     np.save(path / 'res_cube.npy', res_fusion.x)
    #     np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val_lcg)
    # else:
    #     print("Templates loaded, save templates and cube")
    #     np.save(path / 'res_x.npy', res_fusion.x)
    #     np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val_lcg)
    #     np.save(path / 'res_cube.npy', spectroModel.mapsToCube(np.array(res_fusion.x)))

    cube = maps_to_cube(res_fusion.x, L_specs)


    # Choix des bandes spectrales à afficher
    band_indices = [0, 100, 300, 500, 800, 1000]

    n_bands = len(band_indices)
    ncols = 3  # Ground truth, reconstruit, erreur
    nrows = n_bands

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))

    for i, band in enumerate(band_indices):
        # Ground truth
        im0 = axes[i, 0].imshow(ground_truth[band], cmap='gray')
        axes[i, 0].set_title(f"Vrai - Bande {band}")
        axes[i, 0].axis('off')
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Reconstitué
        im1 = axes[i, 1].imshow(cube[band], cmap='gray')
        axes[i, 1].set_title(f"Reconstruit - Bande {band}")
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Erreur relative
        diff = (ground_truth[band] - cube[band]) / (ground_truth[band] + 1e-8)
        im2 = axes[i, 2].imshow(diff, cmap='seismic')
        axes[i, 2].set_title(f"Erreur relative - Bande {band}")
        axes[i, 2].axis('off')
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle("Comparaison des bandes spectrales : Vrai vs Reconstruit vs Erreur", fontsize=16, y=1.02)
    plt.show()



print("Launch")
data_dir_path = '/home/nmonnier/Data/JWST/INCLASS/IAS_fusion_MIRI/'

noise_path = 'CUBES/30db/'


psfs_monoch = np.load(data_dir_path + "psfs_pixscale0.075_fov2.4.npy")
L_pce_mirim = np.load(data_dir_path + "pce_300.npy")
L_pce_spectro = np.load(data_dir_path + "spectro_pce_300.npy")
lamb_cube = np.load(data_dir_path + "lamb_cube.npy")
ground_truth = np.load(data_dir_path + "GT.npy")

thres = 0.01
L_pce_spectro[L_pce_spectro < thres] = thres


L_specs = (
    np.load(data_dir_path +  noise_path +"MRS_ENDMEMBERS.npy")/ 1e3
)
print(L_specs.shape)

x_old = np.linspace(0, 1, L_pce_mirim.shape[1]) 
x_new = np.linspace(0, 1, 1106) 
interp_func = interp1d(x_old, L_pce_mirim, kind='linear', axis=1, fill_value="extrapolate")
L_pce_mirim = interp_func(x_new)
interp_func = interp1d(x_old, L_pce_spectro, kind='linear', fill_value="extrapolate")
L_pce_spectro = interp_func(x_new)
interp_func = interp1d(x_old, lamb_cube, kind='linear', fill_value="extrapolate")
lamb_cube = interp_func(x_new)



shape_target = (32,32)

# Simplified MRS Subsampling factors 
di = 4
dj = 4


spectro_model = simplifiedMRS.Spectro_Model_Cube(
    psfs_monoch, L_pce_spectro, di, dj, lamb_cube, L_specs, shape_target
)

mirim_model = simplifiedMIRIM.Mirim_Model_LMM(
    psfs_monoch, L_pce_mirim, lamb_cube, L_specs, shape_target
)


data_mrs = np.load(data_dir_path + noise_path + 'MRS.npy')

x_freq = rdft2(data_mrs)
data_mrs_freq = x_freq * rdft2(spectro_model.decal)[np.newaxis, :, :]
data_mrs = irdftn(data_mrs_freq, shape_target)
data_mrs = data_mrs[:,::di, ::dj]
data_mirim = np.load(data_dir_path + noise_path + 'MIRIM.npy')

print("Data mrs = ", data_mrs.shape)
print(spectro_model.ishape)
print(spectro_model.oshape)
# reconstruction_method(mirim_model, data_mirim, spectro_model, data_mrs, None, None, None, None, None)

