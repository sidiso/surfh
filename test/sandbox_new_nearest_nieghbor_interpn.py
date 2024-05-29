import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils
from surfh.Models import instru
from astropy import units as u
from astropy.coordinates import Angle
from numpy.random import standard_normal as randn
from scipy.spatial import cKDTree
from surfh.ToolsDir import nearest_neighbor_interpolation

# def tmp_test_interpolation_FoV2cube_python_cython():
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



gridded = cython_utils.interpn_cube2local(wavel_idx, 
                                                    cube_alpha_axis, 
                                                    cube_beta_axis, 
                                                    cube, 
                                                    optimized_local_coords, 
                                                    local_out_shape)

test_cube_alpha_axis = np.tile(cube_alpha_axis, len(cube_beta_axis))
test_cube_beta_axis= np.repeat(cube_beta_axis, len(cube_beta_axis))

# gridata = griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube[0].ravel(), (local_alpha_coord, local_beta_coord))
# degridata = griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube[0].ravel(), (local_alpha_coord, local_beta_coord))
wavel_indexes = np.tile(indexes, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes)) + ((wavel_idx[...,np.newaxis])*cube[0].size )
gridata = cube.ravel()[wavel_indexes].reshape(len(wavel_idx), local_alpha_coord.shape[0], local_alpha_coord.shape[1])

indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata[0].ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
wavel_indexes_t = np.tile(indexes_t, (len(wavel_idx),1)).reshape(len(wavel_idx), len(indexes_t)) + ((wavel_idx[...,np.newaxis])*local_alpha_coord.shape[0]* local_alpha_coord.shape[1] )
degridata = gridata.ravel()[wavel_indexes_t].reshape(len(wavel_idx), 251,251)

mask = np.zeros_like(cube[0])
mask.ravel()[indexes] = 1
nmask = np.zeros_like(mask)
for i in range(1,cube.shape[1]-1):
    for j in range(1, cube.shape[2]-1):
        if mask[i,j] == 1:
            nmask[i,j] = 1
        else:
            if mask[i-1, j-1] == 1 or mask[i, j-1] == 1 \
                or mask[i+1, j-1] == 1 or mask[i-1, j] == 1\
                or mask[i-1, j+1] == 1 or mask[i+1, j+1] == 1\
                or mask[i, j+1] == 1 or mask[i+1, j] == 1:
                nmask[i,j] = 1

plt.imshow(cube[0]*nmask-degridata*nmask)
