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


def diffcubes(linop: LinOp, num: int = 1, rtol: float = 1e-5, atol: float = 1e-8, echo=False
) -> float:
    vvec = randn(linop.isize)
    uvec = randn(linop.osize)

    left = np.vdot(linop.rmatvec(uvec).ravel(), vvec.ravel())
    right = np.vdot(uvec.ravel(), linop.matvec(vvec).ravel())
    
    diff = left-right
    if echo:
        print(f"(Aᴴ·u)ᴴ·v = {left} ≈ {right} = uᴴ·(A·v)")
    return diff

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

    # Cube Coordinates
    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)
    cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta
    cube_shape = cube.shape


    # Fov Local coordinates
    angle = 45
    local_alpha_axis = np.arange(local_shape[0]).astype(np.float64)
    local_beta_axis = np.arange(local_shape[1]).astype(np.float64)
    test_local_alpha_axis = local_alpha_axis.copy()
    test_local_beta_axis = local_beta_axis.copy()
    test_local_alpha_axis = np.tile(test_local_alpha_axis, local_shape[1])
    test_local_beta_axis = np.repeat(test_local_beta_axis, local_shape[0])
    A =  np.vstack((test_local_alpha_axis, test_local_beta_axis))
    theta = -np.pi / 5
    rotate = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    translate = A.mean(axis=1, keepdims=True)
    tmp_coords = A - translate
    tmp_coords = rotate @ tmp_coords
    
    local_in_global_alpha_axis = tmp_coords[0].reshape((local_shape[0], local_shape[1]))
    local_in_global_beta_axis  = tmp_coords[1].reshape((local_shape[0], local_shape[1]))


    # local_alpha_axis -= np.mean(local_alpha_axis)
    # local_beta_axis -= np.mean(local_beta_axis)

    # # Transform local coordinates in Cube coordinates system
    # local_in_global_alpha_axis_tmp = np.tile(local_alpha_axis.reshape((-1, 1)), [1, local_shape[0]])
    # local_in_global_beta_axis_tmp = np.tile(local_beta_axis.reshape((1, -1)), [local_shape[1], 1])
    # coords = utils.rotmatrix(angle) @ np.vstack(
    #         (local_in_global_alpha_axis_tmp.ravel(), local_in_global_beta_axis_tmp.ravel())
    #     )
    
    # local_in_global_alpha_axis = coords[0].reshape((local_shape[0], local_shape[1])) + cube_origin_alpha
    # local_in_global_beta_axis  = coords[1].reshape((local_shape[0], local_shape[1])) + cube_origin_beta

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

    
    cython_interpn = cython_utils.interpn_cube2local(wavel_idx, 
                                                     cube_alpha_axis, 
                                                     cube_beta_axis, 
                                                     cube, 
                                                     optimized_local_coords, 
                                                     local_cube_shape)
    


    # Cube axis in FoV coordinates
    tmp_cube_alpha_axis = np.tile(cube_alpha_axis, cube_shape[2])
    tmp_cube_beta_axis = np.repeat(cube_beta_axis, cube_shape[1])
    B =  np.vstack((tmp_cube_alpha_axis, tmp_cube_beta_axis))
    inv_rotate = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta),  np.cos(-theta)]
    ])
    translate = B.mean(axis=1, keepdims=True)
    tmp_coords = B - translate
    tmp_coords = inv_rotate @ tmp_coords

    global_alpha_axis = tmp_coords[0].reshape((cube_shape[1], cube_shape[2]))
    global_beta_axis  = tmp_coords[1].reshape((cube_shape[1], cube_shape[2]))

    # FoV -> Cube
    global_cube_alpha_axis = np.tile(cube_alpha_axis.reshape((-1, 1)), [1, cube_shape[1]])
    global_cube_beta_axis = np.tile(cube_beta_axis.reshape((1, -1)), [cube_shape[2], 1])
    coords = np.vstack(
            (global_cube_alpha_axis.ravel(), global_cube_beta_axis.ravel())
        )

    global_coords = np.vstack(
            [
                np.tile(
                    wavel_idx.reshape((-1, 1, 1)), (1, cube_shape[1], cube_shape[2])
                ).ravel(),
                np.repeat(global_alpha_axis[np.newaxis, np.newaxis], cube_shape[0], axis=0).ravel(),
                np.repeat(global_beta_axis[np.newaxis, np.newaxis], cube_shape[0], axis=0).ravel(),
            ]
        ).T

    print(local_in_global_alpha_axis[:,0].ravel().shape)
    print(local_in_global_beta_axis[0].ravel().shape)
    points = (wavel_idx, local_in_global_alpha_axis[:,0].ravel(), local_in_global_beta_axis[0].ravel())
    for i, p in enumerate(points):
        if not np.all(np.diff(p) > 0.):
            print(p)
            print("hhhhhhhhhhhh")
    print("local_alpha_axis", local_beta_axis)
    print(local_in_global_beta_axis[:,0].ravel())
    out = python_utils.interpn_local2cube(wavel_idx, 
                                            local_in_global_alpha_axis[:,0].ravel(), 
                                            local_in_global_beta_axis[:,0].ravel(), 
                                            cython_interpn, 
                                            global_coords, 
                                            cube_shape)
    
    plt.plot(local_in_global_alpha_axis[0,0],  local_in_global_beta_axis[0,0] , '.')
    plt.plot(local_in_global_alpha_axis[0,0],  local_in_global_beta_axis[0,-1], '.')
    plt.plot(local_in_global_alpha_axis[-1,0], local_in_global_beta_axis[0,0] , '.')
    plt.plot(local_in_global_alpha_axis[-1,0], local_in_global_beta_axis[0,-1], '.')
    plt.plot(local_in_global_alpha_axis[0,0],  local_in_global_beta_axis[0,0] , '.')

    plt.plot(cube_alpha_axis[0], cube_beta_axis[0], 'x')
    plt.plot(cube_alpha_axis[0], cube_beta_axis[-1], 'x')
    plt.plot(cube_alpha_axis[-1],cube_beta_axis[0], 'x')
    plt.plot(cube_alpha_axis[-1],cube_beta_axis[-1], 'x')
    plt.plot(cube_alpha_axis[0], cube_beta_axis[0], 'x')

    plt.plot(tmp_coords[0], tmp_coords[1], '.')

    plt.show()
    print(tmp_coords.shape)

    return out, cython_interpn, cube




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


methodfw = 'linear'
methodad = 'linear'
gridata = sp.interpolate.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube[0].ravel(), (local_alpha_coord, local_beta_coord), method=methodfw)
degridata = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)

# mask = np.where(degridata< 1e-1, True, False)

# mask_coord = np.where(degridata> 1e-1)
# cube_masked = degridata[mask_coord]

# gridata2 = sp.interpolate.griddata((cube_alpha_axis[mask_coord[0].ravel()], cube_beta_axis[mask_coord[1].ravel()]), cube_masked.ravel(), (local_alpha_coord, local_beta_coord), method=methodfw, fill_value=0)
# degridata2 = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata2.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)

mask_coord = np.where(degridata< 1e-1)
cube2 = cube.copy()
cube2[0][np.where(degridata< 1e-1)] = 0
gridata = sp.interpolate.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube2[0].ravel(), (local_alpha_coord, local_beta_coord), method=methodfw)
degridata1 = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)

degridata1[np.where(degridata< 1e-1)] = 0
gridata2   = sp.interpolate.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), degridata1.ravel(), (local_alpha_coord, local_beta_coord), method=methodfw)
degridata2 = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata2.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)
degridata2[np.where(degridata< 1e-1)] = 0

gridata3   = sp.interpolate.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), degridata2.ravel(), (local_alpha_coord, local_beta_coord), method=methodfw)
degridata3 = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata3.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)
degridata3[np.where(degridata< 1e-1)] = 0

gridata4   = sp.interpolate.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), degridata3.ravel(), (local_alpha_coord, local_beta_coord), method=methodfw)
degridata4 = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata4.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)
degridata4[np.where(degridata< 1e-1)] = 0

gridata5   = sp.interpolate.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), degridata4.ravel(), (local_alpha_coord, local_beta_coord), method=methodfw)
degridata5 = sp.interpolate.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata5.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)), method=methodad, fill_value=0)
degridata5[np.where(degridata< 1e-1)] = 0

utils.plot_3_cube(degridata1[np.newaxis,...], degridata5[np.newaxis,...], (degridata4-degridata5)[np.newaxis,...], slice=0)
plt.show()

################################""""



alpha_coord, beta_coord = (ch2a.fov).global2local(
        cube_alpha_axis, cube_beta_axis
    )
cube_out_shape = (len(wavel_idx), len(cube_alpha_axis), len(cube_beta_axis))
global_coords = np.vstack(
        [
            alpha_coord.ravel(),
            beta_coord.ravel()
        ]
    ).T

degridded = cython_utils.interpn_local2cube(wavel_idx, 
                                        local_alpha_axis.ravel(), 
                                        local_beta_axis.ravel(), 
                                        gridded, 
                                        global_coords, 
                                        cube_out_shape)

print(gridded.shape)

# return cube, gridded, p_gridded, degridded












gridded2 = cython_utils.interpn_cube2local(wavel_idx, 
                                            cube_alpha_axis, 
                                            cube_beta_axis, 
                                            degridded, 
                                            optimized_local_coords, 
                                            local_out_shape)

degridded2 = cython_utils.interpn_local2cube(wavel_idx, 
                                        local_alpha_axis.ravel(), 
                                        local_beta_axis.ravel(), 
                                        gridded2, 
                                        global_coords, 
                                        cube_out_shape)

# maps = global_variable_testing.maps
# templates = global_variable_testing.templates
# im_shape = global_variable_testing.im_shape
# wavelength_axis = global_variable_testing.wavelength_axis
# lmm_class = LMM(maps, templates, wavelength_axis)
# n_lamnda = len(global_variable_testing.wavelength_axis)
# cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))
# f_cube = python_utils.dft(cube)

# dt = dottest(lmm_class)


# _, _, _ = test_interpolation_cubeFoV_python_cython()
# _, _, _ = test_interpolation_FoV2cube_python_cython()

# cube, gridded, p_gridded, degridded = tmp_test_interpolation_FoV2cube_python_cython()

# plt.figure()
# plt.imshow(cube[0])
# plt.colorbar()

# plt.figure()
# plt.imshow(gridded[0])
# plt.colorbar()

# plt.figure()
# plt.imshow(p_gridded[0])
# plt.colorbar()

# plt.show()



# def test_interpn_python_dottest():
#     """
#     Model : y = Sx

#     y : Hyperspectral cube of size (L, Nx, Ny)
#     S : Interpolation operator
#     x : Hyperspectral cube of size (4, Nx, Ny)
#     """
#     from numpy import ndarray as array
#     class Interpn(LinOp):
#         def __init__(
#                     self,
#                     sotf: array,
#                     templates: array,
#                     alpha_axis: array,
#                     beta_axis: array,
#                     wavelength_axis: array,
#                     instr: instru.IFU,  
#                     step_Angle: float
#                     ):
#             self.sotf = sotf
#             self.templates = templates
#             self.alpha_axis = alpha_axis
#             self.beta_axis = beta_axis
#             self.wavelength_axis = wavelength_axis
#             self.inst = instr
#             self.step_Angle = step_Angle

#             local_alpha_axis, local_beta_axis = self.inst.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
            
#             self.local_alpha_axis = local_alpha_axis
#             self.local_beta_axis = local_beta_axis

#             ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
#             oshape = (len(self.wavelength_axis), len(local_alpha_axis), len(local_beta_axis))


#             super().__init__(ishape=ishape, oshape=oshape)

#         def forward(self, cube: np.ndarray) -> np.ndarray:
#             local_alpha_coord, local_beta_coord = self.inst.fov.local2global(
#                                                         self.local_alpha_axis, self.local_beta_axis
#                                                         )
#             optimized_local_coords = np.vstack(
#                                             [
#                                                 local_alpha_coord.ravel(),
#                                                 local_beta_coord.ravel()
#                                             ]
#                                             ).T 
#             return cython_utils.interpn_cube2local(self.wavelength_axis, 
#                                                    self.alpha_axis, 
#                                                    self.beta_axis, 
#                                                    cube, 
#                                                    optimized_local_coords, 
#                                                    self.oshape)
        
#         def adjoint(self, fov: np.ndarray) -> np.ndarray:
            
#             alpha_coord, beta_coord = self.inst.fov.global2local(
#                     self.alpha_axis, self.beta_axis
#                     )
            
#             optimized_global_coords = np.vstack(
#                 [
#                     alpha_coord.ravel(),
#                     beta_coord.ravel()
#                 ]
#                 ).T
            
#             return cython_utils.interpn_local2cube(self.wavelength_axis, 
#                                                    self.local_alpha_axis.ravel(), 
#                                                    self.local_beta_axis.ravel(), 
#                                                    fov, 
#                                                    optimized_global_coords, 
#                                                    self.ishape)

#     maps = global_variable_testing.maps
#     templates = global_variable_testing.templates
#     im_shape = global_variable_testing.im_shape
#     wavelength_axis = global_variable_testing.wavelength_axis
#     sotf = global_variable_testing.sotf

#     step = 0.025 # arcsec
#     step_Angle = Angle(step, u.arcsec)

#     cube_origin_alpha = 0
#     cube_origin_beta = 0
#     cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
#     cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
#     cube_alpha_axis -= np.mean(cube_alpha_axis)
#     cube_beta_axis -= np.mean(cube_beta_axis)
#     cube_alpha_axis += cube_origin_alpha
#     cube_beta_axis += cube_origin_beta

    

#     # Def Channel spec.
#     ch2a = instru.IFU(
#         fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
#         det_pix_size=0.196,
#         n_slit=17,
#         w_blur=None,
#         pce=None,
#         wavel_axis=None,
#         name="2A",
#     )

#     interpn_class = Interpn(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, ch2a, step_Angle)
#     lmm_class = LMM(maps, templates, wavelength_axis)
#     return diffcubes(lmm_class)

# print(test_interpn_python_dottest())