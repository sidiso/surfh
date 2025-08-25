import pathlib
import numpy as np

from surfh.Simulation.fusion_CT import QuadCriterion_MRS
from surfh.Vizualisation import cube_vizualisation




# Main function to execute the reconstruction method
def reconstruction_method(spectroModel, ndata, templates, result_path, hyperParameter, niter, method, scale_data):
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
    result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_4_Temp_{shape_templates}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}_SD_{scale_data}/'
    path = pathlib.Path(result_path + result_dir)
    path.mkdir(parents=True, exist_ok=True)

    # QuadCriterion initialization
    quadCrit_fusion = QuadCriterion_MRS(
        mu_spectro=1,
        y_spectro=np.copy(ndata),
        model_spectro=spectroModel,
        mu_reg=hyperParameter,
        printing=True,
        gradient="separated"
    )

    # Run the method
    res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit=1, calc_crit=True, value_init=value_init)

    if templates is None:
        print("No templates")
        print(f"Results save in {path}")
        np.save(path / 'res_cube.npy', res_fusion.x)
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)
    else:
        # Convert maps to cube
        y_cube = spectroModel.mapsToCube(res_fusion.x)

        # Save results
        print(f"Results save in {path}")
        np.save(path / 'res_x.npy', res_fusion.x)
        np.save(path / 'res_cube.npy', y_cube)
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val)

    
    # cube_vizualisation.plot_cube(y_cube, spectroModel.wavelength_axis)