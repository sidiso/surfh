import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing

from astropy import units as u
from astropy.coordinates import Angle

from surfh.ToolsDir import jax_utils, python_utils, cython_utils
from surfh.Models import instru, slicer, newslicer

from typing import List, Tuple
from numpy import ndarray as array



def test_LMM_python_dottest():
    """
    Model : y = Ta

    y : Hyperspectral cube of size (L, Nx, Ny)
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
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

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            return python_utils.lmm_maps2cube(maps, self.templates)
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            return python_utils.lmm_cube2maps(cube, self.templates)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    lmm_class = LMM(maps, templates, wavelength_axis)

    assert dottest(lmm_class)

def test_LMM_jax_dottest():
    """
    Model : y = Ta

    y : Hyperspectral cube of size (L, Nx, Ny)
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
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

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            maps = maps.reshape(self.ishape)
            return jax_utils.lmm_maps2cube(maps, self.templates).reshape(self.oshape)
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            cube = cube.reshape(self.oshape)
            return jax_utils.lmm_cube2maps(cube, self.templates).reshape(self.ishape)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    lmm_class = LMM(maps, templates, wavelength_axis)

    assert dottest(lmm_class)



def test_spatial_conv_python_dottest():
    """
    Model : y = Cx

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    """
    class Conv(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.sotf = sotf
            self.maps = maps

            ishape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, inarray: np.ndarray) -> np.ndarray:
            return python_utils.idft(python_utils.dft(inarray) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, inarray: np.ndarray) -> np.ndarray:
            return python_utils.idft(python_utils.dft(inarray) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf
    conv_class = Conv(sotf, maps, templates, wavelength_axis)

    assert dottest(conv_class)


def test_spatial_conv_jax_dottest():
    """
    Model : y = Cx

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    """
    class Conv(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.sotf = sotf
            self.maps = maps

            ishape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, inarray: np.ndarray) -> np.ndarray:
            return jax_utils.idft(jax_utils.dft(inarray) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, inarray: np.ndarray) -> np.ndarray:
            return jax_utils.idft(jax_utils.dft(inarray) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf
    conv_class = Conv(sotf, maps, templates, wavelength_axis)

    assert dottest(conv_class)


def test_LMMConv_python_dottest():
    """
    Model : y = CTa

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    class LMMConv(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.sotf = sotf
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.templates = templates # ex taille : [4, 307]
            self.maps = maps # ex taille : [4, 251, 251]

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            cube = python_utils.lmm_maps2cube(maps, self.templates)
            return python_utils.idft(python_utils.dft(cube) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            conv_cube = python_utils.idft(python_utils.dft(cube) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))
            return python_utils.lmm_cube2maps(conv_cube, self.templates)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    lmmConv_class = LMMConv(sotf, maps, templates, wavelength_axis)

    assert dottest(lmmConv_class)

def test_LMMConv_jax_dottest():
    """
    Model : y = CTa

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
    class LMMConv(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.sotf = sotf
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.templates = templates # ex taille : [4, 307]
            self.maps = maps # ex taille : [4, 251, 251]

            ishape = (4, maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, maps: np.ndarray) -> np.ndarray:
            cube = jax_utils.lmm_maps2cube(maps, self.templates)
            return jax_utils.idft(jax_utils.dft(cube) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, cube: np.ndarray) -> np.ndarray:
            conv_cube = jax_utils.idft(jax_utils.dft(cube) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))
            return jax_utils.lmm_cube2maps(conv_cube, self.templates)


    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    lmmConv_class = LMMConv(sotf, maps, templates, wavelength_axis)

    assert dottest(lmmConv_class)


def test_interpn_python_dottest():
    """
    Model : y = STx

    y : Hyperspectral cube of size (L, Nx, Ny)
    S : Interpolation operator
    T : LMM transformation operator
    x : Hyperspectral cube of size (4, Nx, Ny)
    """
    class Interpn(LinOp):
        def __init__(
                    self,
                    sotf: array,
                    templates: array,
                    alpha_axis: array,
                    beta_axis: array,
                    wavelength_axis: array,
                    instr: instru.IFU,  
                    step_Angle: float
                    ):
            self.sotf = sotf
            self.templates = templates
            self.alpha_axis = alpha_axis
            self.beta_axis = beta_axis
            self.wavelength_axis = wavelength_axis
            self.inst = instr
            self.step_Angle = step_Angle

            local_alpha_axis, local_beta_axis = self.inst.fov.local_coords(step_Angle.degree, 5* step_Angle.degree, 5* step_Angle.degree)
            
            self.local_alpha_axis = local_alpha_axis
            self.local_beta_axis = local_beta_axis

            ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
            oshape = (len(self.wavelength_axis), len(local_alpha_axis), len(local_beta_axis))


            super().__init__(ishape=ishape, oshape=oshape)

        def forward(self, cube: np.ndarray) -> np.ndarray:
            local_alpha_coord, local_beta_coord = self.inst.fov.local2global(
                                                        self.local_alpha_axis, self.local_beta_axis
                                                        )
            optimized_local_coords = np.vstack(
                                            [
                                                local_alpha_coord.ravel(),
                                                local_beta_coord.ravel()
                                            ]
                                            ).T 
            return cython_utils.interpn_cube2local(self.wavelength_axis, 
                                                   self.alpha_axis, 
                                                   self.beta_axis, 
                                                   cube, 
                                                   optimized_local_coords, 
                                                   self.oshape)
        
        def adjoint(self, fov: np.ndarray) -> np.ndarray:
            
            alpha_coord, beta_coord = self.inst.fov.global2local(
                    self.alpha_axis, self.beta_axis
                    )
            
            optimized_global_coords = np.vstack(
                [
                    alpha_coord.ravel(),
                    beta_coord.ravel()
                ]
                ).T
            
            return cython_utils.interpn_local2cube(self.wavelength_axis, 
                                                   self.local_alpha_axis.ravel(), 
                                                   self.local_beta_axis.ravel(), 
                                                   fov, 
                                                   optimized_global_coords, 
                                                   self.ishape)

    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta

    

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

    interpn_class = Interpn(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, ch2a, step_Angle)

    assert dottest(interpn_class)



def test_slicing_python_dottest():
    """
    Model : y = STx

    y : Hyperspectral cube of size (L, Sx, Sy)
    S : Slicing operation
    T : LMM transformation operator
    x : Hyperspectral cube of size (4, Nx, Ny)
    """
    class SlicerModel(LinOp):
        def __init__(
                    self,
                    sotf: array,
                    templates: array,
                    alpha_axis: array,
                    beta_axis: array,
                    wavelength_axis: array,
                    instr: instru.IFU,  
                    step_degree: float
                    ):
            self.sotf = sotf
            self.templates = templates
            self.alpha_axis = alpha_axis
            self.beta_axis = beta_axis
            self.wavelength_axis = wavelength_axis
            self.instr = instr
            self.step_degree = step_degree

            local_alpha_axis, local_beta_axis = self.instr.fov.local_coords(step_degree, 0, 0)#5* step_degree, 5* step_degree)
            
            self.local_alpha_axis = local_alpha_axis
            self.local_beta_axis = local_beta_axis

            self.slicer = slicer.Slicer(self.instr, 
                                        wavelength_axis = self.wavelength_axis, 
                                        alpha_axis = self.alpha_axis, 
                                        beta_axis = self.beta_axis, 
                                        local_alpha_axis = self.local_alpha_axis, 
                                        local_beta_axis = self.local_beta_axis)
            
            self.newslicer = newslicer.newSlicer(self.instr, 
                                        wavelength_axis = self.wavelength_axis, 
                                        alpha_axis = self.alpha_axis, 
                                        beta_axis = self.beta_axis, 
                                        local_alpha_axis = self.local_alpha_axis, 
                                        local_beta_axis = self.local_beta_axis)

            # Templates (4, Nx, Ny)
            ishape = (len(self.wavelength_axis), len(alpha_axis), len(beta_axis))
            
            # 4D array [Nslit, L, alpha_slit, beta_slit]
            oshape = (self.instr.n_slit, len(self.wavelength_axis), self.newslicer.npix_slit_alpha_width, self.slicer.npix_beta_slit)


            super().__init__(ishape=ishape, oshape=oshape)


        @property
        def n_alpha(self) -> int:
            """The number of input pixel inside a slit (along slit dim)"""
            return self.instr.fov.local.n_alpha(self.step_degree)

        def forward(self, cube: array) -> array:
            allsliced = np.zeros(self.oshape)
            for slit_idx in range(self.instr.n_slit):
                sliced = self.newslicer.slicing(cube, slit_idx)
                allsliced[slit_idx] = sliced
            return allsliced


        def adjoint(self, slices: array) -> array:
            cube = np.zeros((len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
            for slit_idx in range(self.instr.n_slit):
                sliced = slices[slit_idx]
                cube += self.newslicer.slicing_t(sliced, slit_idx, (len(self.wavelength_axis), self.ishape[1], self.ishape[2]))
            return cube

    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta

    alpha_fov_rchan = cube_alpha_axis[-1] - cube_alpha_axis[0]
    beta_fov_rchan = cube_beta_axis[-1] - cube_beta_axis[0]

    alpha_width_fov_rchan = (cube_alpha_axis[-1] - cube_alpha_axis[0])# - step_Angle.degree*10
    beta_width_fov_rchan = (cube_beta_axis[-1] - cube_beta_axis[0])# - step_Angle.degree*10

    # Def random instru spec.
    rchan = instru.IFU(
        fov=instru.FOV(alpha_width_fov_rchan, beta_width_fov_rchan, origin=instru.Coord(0, 0), angle=0),
        det_pix_size=0.196,
        n_slit=17,
        w_blur=None,
        pce=None,
        wavel_axis=wavelength_axis,
        name="2A"
    )

    sliceModel = SlicerModel(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)
    dottest(sliceModel)


def test_spectral_conv_python_dottest():
    """
    Model : y = Wx

    y : Hyperspectral cube of size (L', Nx, Ny)
    W : Spectral blur operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    
    # [λ', α, β] = ∑_λ arr[λ, α, β] wpsf[λ', λ, β]
    # Σ_λ
    """
    class WBlur(LinOp):
        def __init__(
            self,
            sotf,
            maps,
            templates,
            wavelength_axis  
        ):
            self.wavelength_axis = wavelength_axis # ex taille [307]
            self.sotf = sotf
            self.maps = maps

            ishape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])#maps.shape
            oshape = (len(self.wavelength_axis), maps.shape[1], maps.shape[2])
            super().__init__(ishape=ishape, oshape=oshape)
            print(self.ishape, self.oshape)


        def forward(self, inarray: np.ndarray) -> np.ndarray:
            return python_utils.idft(python_utils.dft(inarray) * self.sotf, (self.maps.shape[1], self.maps.shape[2]))
        
        def adjoint(self, inarray: np.ndarray) -> np.ndarray:
            return python_utils.idft(python_utils.dft(inarray) * self.sotf.conj(), (self.maps.shape[1], self.maps.shape[2]))

    pass