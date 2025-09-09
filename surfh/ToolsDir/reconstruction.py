import pathlib
import numpy as np

from surfh.Simulation.fusion_CT import QuadCriterion_MRS
from surfh.Vizualisation import cube_vizualisation
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits




# Main function to execute the reconstruction method
def reconstruction_method(MRSModel, ndata, templates, result_path, hyperParameter, niter, method, scale_data, data_dict):
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

    # Create result directory
    if templates is None:
        shape_templates = 'None'
    else:
        shape_templates = templates.shape[0]
    result_dir = f'{method}_MC_{len(MRSModel.instrs)}_MO_4_Temp_{shape_templates}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}_SD_{scale_data}/'
    path = pathlib.Path(result_path + result_dir)
    path.mkdir(parents=True, exist_ok=True)

    # QuadCriterion initialization
    quadCrit_fusion = QuadCriterion_MRS(
        mu_spectro=1,
        y_spectro=np.copy(ndata),
        model_spectro=MRSModel,
        mu_reg=hyperParameter,
        printing=True,
        gradient="separated"
    )

    # Run the method
    res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit=1, calc_crit=True, value_init=value_init)

    metadata = {'PA_V3': data_dict['PA_V3']['1c'], 
                'TARG_RA': data_dict['target']['1c'][0], 'TARG_DEC':data_dict['target']['1c'][1], 
                'RA_V1': data_dict['targetV1']['1c'][0], 'DEC_V1': data_dict['targetV1']['1c'][1], 
                'RA_REF': data_dict['targetREF']['1c'][0], 'DEC_REF': data_dict['targetREF']['1c'][1],
                'ALPHA_AXIS':MRSModel.alpha_axis, 'BETA_AXIS':MRSModel.beta_axis, 'WAVELENGTH': MRSModel.wavelength_axis}
    for key in metadata.keys():
        print(f"{key}: {metadata[key]}")

    if templates is None:
        print("No templates")
        print(f"Results save in {path}")
        # np.save(path / 'res_cube.npy', res_fusion.x)
        save_numpy_to_fits(np.array(res_fusion.x), metadata, path /'res_cube.fits')
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)
    else:
        # Convert maps to cube
        y_cube = MRSModel.mapsToCube(res_fusion.x)

        # Save results
        print(f"Results save in {path}")
        np.save(path / 'res_x.npy', res_fusion.x)
        # np.save(path / 'res_cube.npy', y_cube)
        save_numpy_to_fits(np.array(y_cube), metadata, path/'res_cube.fits')
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)
        np.save(path / 'wavel.npy', MRSModel.wavelength_axis)

    # cube_vizualisation.plot_cube(y_cube, spectroModel.wavelength_axis)