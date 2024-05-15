import pytest
import numpy as np

from surfh.Models import instru
from astropy import units as u
from astropy.coordinates import Angle
from scipy import misc

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
def test_interpolation_cube2FoV_python_cython():
    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    local_shape = (im_shape[0]-100, im_shape[1]-100)
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))
    out_cube = np.zeros_like(cube)

    face = misc.face()
    face = face[::2,::4,:]
    face = face[:im_shape[0], :im_shape[1], 0]
    cube[:] = face

    # Wavelength index
    wavel_idx = np.arange(n_lamnda)

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    # Cube Coordinates
    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta
    cube_shape = cube.shape

    # Def Channel spec.
    ch2a = instru.IFU(
        fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
        det_pix_size=0.196,
        n_slit=17,
        w_blur=None,
        pce=None,
        wavel_axis=None,
        name="2A",
    )

    local_alpha_axis, local_beta_axis = ch2a.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
    local_alpha_coord, local_beta_coord = (ch2a.fov).local2global(
            local_alpha_axis, local_beta_axis
        )
    local_out_shape = (len(wavel_idx),) + local_alpha_coord.shape


    local_coords = np.vstack(
            [
                np.repeat(
                    np.repeat(wavel_idx.reshape((-1, 1, 1)), local_out_shape[1], axis=1),
                    local_out_shape[2],
                    axis=2,
                ).ravel(),
                np.repeat(local_alpha_coord[np.newaxis], local_out_shape[0], axis=0).ravel(),
                np.repeat(local_beta_coord[np.newaxis], local_out_shape[0], axis=0).ravel(),
            ]
        ).T
    optimized_local_coords = np.vstack(
            [
                local_alpha_coord.ravel(),
                local_beta_coord.ravel()
            ]
        ).T 

    p_gridded = python_utils.interpn_cube2local(wavel_idx, 
                                                     cube_alpha_axis, 
                                                     cube_beta_axis, 
                                                     cube, 
                                                     local_coords, 
                                                     local_out_shape)
    
    c_gridded = cython_utils.interpn_cube2local(wavel_idx, 
                                                     cube_alpha_axis, 
                                                     cube_beta_axis, 
                                                     cube, 
                                                     optimized_local_coords, 
                                                     local_out_shape)

    assert np.allclose(p_gridded, c_gridded)

"""
Interpolation - FoV -> cube
"""
def test_interpolation_FoV2cube_python_cython():
    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    local_shape = (im_shape[0]-100, im_shape[1]-100)
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))
    out_cube = np.zeros_like(cube)

    face = misc.face()
    face = face[::2,::4,:]
    face = face[:im_shape[0], :im_shape[1], 0]
    cube[:] = face

    # Wavelength index
    wavel_idx = np.arange(n_lamnda)

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    # Cube Coordinates
    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta
    cube_shape = cube.shape

    # Def Channel spec.
    ch2a = instru.IFU(
        fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
        det_pix_size=0.196,
        n_slit=17,
        w_blur=None,
        pce=None,
        wavel_axis=None,
        name="2A",
    )

    local_alpha_axis, local_beta_axis = ch2a.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
    local_alpha_coord, local_beta_coord = (ch2a.fov).local2global(
            local_alpha_axis, local_beta_axis
        )
    local_out_shape = (len(wavel_idx),) + local_alpha_coord.shape

    optimized_local_coords = np.vstack(
            [
                local_alpha_coord.ravel(),
                local_beta_coord.ravel()
            ]
        ).T 

    
    c_gridded = cython_utils.interpn_cube2local(wavel_idx, 
                                                     cube_alpha_axis, 
                                                     cube_beta_axis, 
                                                     cube, 
                                                     optimized_local_coords, 
                                                     local_out_shape)
    
    print(c_gridded.shape)
    alpha_coord, beta_coord = (ch2a.fov).global2local(
            cube_alpha_axis, cube_beta_axis
        )
    cube_out_shape = (len(wavel_idx), len(cube_alpha_axis), len(cube_beta_axis))

    global_coords = np.vstack(
            [
                np.tile(
                    wavel_idx.reshape((-1, 1, 1)), (1, cube_out_shape[1], cube_out_shape[2])
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], cube_out_shape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], cube_out_shape[0], axis=0).ravel(),
            ]
        ).T

    optimized_global_coords = np.vstack(
            [
                alpha_coord.ravel(),
                beta_coord.ravel()
            ]
        ).T

    c_degridded = python_utils.interpn_local2cube(wavel_idx, 
                                            local_alpha_axis.ravel(), 
                                            local_beta_axis.ravel(), 
                                            c_gridded, 
                                            global_coords, 
                                            cube_out_shape)

    c_degridded = cython_utils.interpn_local2cube(wavel_idx, 
                                            local_alpha_axis.ravel(), 
                                            local_beta_axis.ravel(), 
                                            c_gridded, 
                                            optimized_global_coords, 
                                            cube_out_shape)

    assert np.allclose(c_degridded, c_degridded)