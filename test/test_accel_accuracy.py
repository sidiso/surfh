import pytest
import numpy as np
import global_variable_testing

from surfh.ToolsDir import jax_utils, python_utils


def test_LMM_mapsToCube():

    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    
    assert np.allclose(python_utils.lmm_maps2cube(maps, templates), jax_utils.lmm_maps2cube(maps, templates), rtol=1e-5)


def test_LMM_cubeToMaps():

    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))

    assert np.allclose(python_utils.lmm_cube2maps(cube, templates), jax_utils.lmm_cube2maps(cube, templates), rtol=1e-5)

