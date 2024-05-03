import pytest
import numpy as np
import global_variable_testing


from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils

"""
LMM - maps -> cube 
"""
def test_LMM_mapsToCube_python_jax():

    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    
    assert np.allclose(python_utils.lmm_maps2cube(maps, templates), jax_utils.lmm_maps2cube(maps, templates), rtol=1e-5)

"""
LMM - cube -> maps 
"""
def test_LMM_cubeToMaps_python_jax():

    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))

    assert np.allclose(python_utils.lmm_cube2maps(cube, templates), jax_utils.lmm_cube2maps(cube, templates), rtol=1e-5)

"""
DFT
"""
def test_dft_python_jax():
    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))

    assert np.allclose(python_utils.dft(cube), jax_utils.dft(cube), rtol=1e-2)

"""
iDFT
"""
def test_idft_python_jax():
    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))
    f_cube = python_utils.dft(cube)

    assert np.allclose(python_utils.idft(f_cube, im_shape), jax_utils.idft(f_cube, im_shape), atol=1e-6)

"""
Interpolation - cube -> FoV
"""
def test_interpolation_cubeFoV_python_cython():
    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    local_shape = (im_shape[0]-40, im_shape[1]-40)
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))


    # Wavelength index
    wavel_idx = np.arange(n_lamnda)

    # Cube Coordinates
    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)
    cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta


    # Fov Local coordinates
    angle = 8.7
    local_alpha_axis = np.arange(local_shape[0]).astype(np.float64)
    local_beta_axis = np.arange(local_shape[1]).astype(np.float64)
    local_alpha_axis -= np.mean(local_alpha_axis)
    local_beta_axis -= np.mean(local_beta_axis)

    # Transform local coordinates in Cube coordinates system
    local_in_global_alpha_axis = np.tile(local_alpha_axis.reshape((-1, 1)), [1, local_shape[0]])
    local_in_global_beta_axis = np.tile(local_beta_axis.reshape((1, -1)), [local_shape[1], 1])
    coords = utils.rotmatrix(angle) @ np.vstack(
            (local_in_global_alpha_axis.ravel(), local_in_global_beta_axis.ravel())
        )
    
    print(local_alpha_axis.shape,local_beta_axis.shape)
    print(local_in_global_alpha_axis.shape)
    print(local_shape)
    local_in_global_alpha_axis = coords[0].reshape((local_shape[0], local_shape[1])) + cube_origin_alpha
    local_in_global_beta_axis  = coords[1].reshape((local_shape[0], local_shape[1])) + cube_origin_beta

    local_cube_shape = (len(wavel_idx),) + local_shape
    local_coords = np.vstack(
            [
                np.repeat(
                    np.repeat(wavel_idx.reshape((-1, 1, 1)), local_cube_shape[1], axis=1),
                    local_cube_shape[2],
                    axis=2,
                ).ravel(),
                np.repeat(local_in_global_alpha_axis[np.newaxis], local_cube_shape[0], axis=0).ravel(),
                np.repeat(local_in_global_beta_axis[np.newaxis], local_cube_shape[0], axis=0).ravel(),
            ]
        ).T
    optimized_local_coords = np.vstack(
            [
                local_in_global_alpha_axis.ravel(),
                local_in_global_beta_axis.ravel()
            ]
        ).T
    
    interpolate_python = python_utils.interpn_cube2local(wavel_idx, 
                                                         cube_alpha_axis, 
                                                         cube_beta_axis,
                                                         cube,
                                                         local_coords,
                                                         local_cube_shape)
    
    cython_interpn = cython_utils.interpn_cube2local(wavel_idx, 
                                                     cube_alpha_axis, 
                                                     cube_beta_axis, 
                                                     cube, 
                                                     optimized_local_coords, 
                                                     local_cube_shape)

    assert np.allclose(interpolate_python, cython_interpn)

"""
Interpolation - FoV -> cube
"""
def test_interpolation_FoVcube_python_cython():
    pass