import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils


class LMM(LinOp):
    def __init__(
        self,
        maps,
        templates,
        wavelength_axis  
    ):
        self.wavelength_axis = wavelength_axis # ex taille [307]
        self.templates = templates # ex taille : [4, 307]
        self.maps = maps # ex taille : [4, 251, 251]

        ishape = (4,251,251)#maps.shape
        oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
        super().__init__(ishape=ishape, oshape=oshape)
        print(self.ishape, self.oshape)


    def forward(self, maps: np.ndarray) -> np.ndarray:
        maps = maps.reshape(self.ishape)
        return python_utils.lmm_maps2cube(maps, self.templates).reshape(self.oshape)
    
    def adjoint(self, cube: np.ndarray) -> np.ndarray:
        cube = cube.reshape(self.oshape)
        return python_utils.lmm_cube2maps(cube, self.templates).reshape(self.ishape)


def test_interpolation_cubeFoV_python_cython():
    templates = global_variable_testing.templates
    n_lamnda = len(global_variable_testing.wavelength_axis)
    im_shape = global_variable_testing.im_shape
    local_shape = (im_shape[0]-80, im_shape[1]-80)
    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))

    face = misc.face()
    face = face[::2,::4,:]
    face = face[:im_shape[0], :im_shape[1], 0]
    cube[:] = face

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
    
    python_interpn = python_utils.interpn_cube2local(wavel_idx, cube_alpha_axis, cube_beta_axis, cube, local_coords, local_cube_shape)
    cython_interpn = cython_utils.interpn_cube2local(wavel_idx, cube_alpha_axis, cube_beta_axis, cube, optimized_local_coords, local_cube_shape)
    return python_interpn, cython_interpn, cube

# maps = global_variable_testing.maps
# templates = global_variable_testing.templates
# im_shape = global_variable_testing.im_shape
# wavelength_axis = global_variable_testing.wavelength_axis
# lmm_class = LMM(maps, templates, wavelength_axis)
# n_lamnda = len(global_variable_testing.wavelength_axis)
# cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))
# f_cube = python_utils.dft(cube)

# dt = dottest(lmm_class)


p_inter, c_inter, cube = test_interpolation_cubeFoV_python_cython()
plt.imshow(p_inter[0])
plt.colorbar()
plt.figure()

plt.imshow(c_inter[0])
plt.colorbar()

plt.figure()
plt.imshow(cube[0])
plt.colorbar()
plt.show()