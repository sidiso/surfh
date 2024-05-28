import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing

from astropy import units as u
from astropy.coordinates import Angle

from surfh.ToolsDir import jax_utils, python_utils, cython_utils
from surfh.Models import instru, slicer

from typing import List, Tuple
from numpy import ndarray as array



def test_LMM_dottest():
    """
    Model : y = Ta

    y : Hyperspectral cube of size (L, Nx, Ny)
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """

    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis

    from surfh.DottestModels import T_Model

    lmm_class = T_Model.T_spectro(maps, templates, wavelength_axis)

    assert dottest(lmm_class)


def test_spatial_conv_dottest():
    """
    Model : y = Cx

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    """
    maps = global_variable_testing.maps
    templates = global_variable_testing.templates
    im_shape = global_variable_testing.im_shape
    wavelength_axis = global_variable_testing.wavelength_axis
    sotf = global_variable_testing.sotf

    from surfh.DottestModels import C_Model

    conv_class = C_Model.C_spectro(sotf, maps, templates, wavelength_axis)

    assert dottest(conv_class)


def test_CT_dottest():
    """
    Model : y = CTa

    y : Hyperspectral cube of size (L, Nx, Ny)
    C : Convolution operator
    T : LMM transformation operator
    a : abondancy maps of size (4, Nx, Ny)
    """
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

    from surfh.DottestModels import CT_Model

    lmmConv_class = CT_Model.CT_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis)

    assert dottest(lmmConv_class)


def test_ST_dottest():
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



def test_LT_dottest():
    """
    Model : y = LTx

    y : Hyperspectral slices of size (Nslices, L', Sx, Sy)
    L : Slicing operation
    T : LMM transformation operator
    x : Hyperspectral cube of size (4, Nx, Ny)
    """

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

    from surfh.DottestModels import LT_Model

    sliceModel = LT_Model.LT_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)
    assert dottest(sliceModel)


def test_R_dottest():
    """
    Model : y = Rx

    y : Hyperspectral cube of size (L', Nx, Ny)
    R : Spectral blur operator
    x : Hyperspectral cube of size (L, Nx, Ny)
    
    # [λ', α, β] = ∑_λ arr[λ, α, β] wpsf[λ', λ, β]
    # Σ_λ
    """
    templates = global_variable_testing.templates
    maps = global_variable_testing.maps
    instr_wavelength_axis = global_variable_testing.wavelength_axis
    n_lamnda = 200
    cube_walength_step = (instr_wavelength_axis[1]-instr_wavelength_axis[0])*2 # Set arbitrary wavel_axis 
    wavelength_axis = np.arange(instr_wavelength_axis[0], instr_wavelength_axis[-1], cube_walength_step)
    n_lamnda = len(wavelength_axis)
    sotf = global_variable_testing.sotf

    im_shape = global_variable_testing.im_shape
    local_shape = (im_shape[0]-100, im_shape[1]-100)

    cube = np.random.random((n_lamnda, im_shape[0]-150, im_shape[1]-150)) # Make smaller cube for easier memory computation
    out_cube = np.zeros_like(cube)

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


    grating_resolution = global_variable_testing.grating_resolution
    spec_blur = global_variable_testing.spec_blur

    # Def Channel spec.
    rchan = instru.IFU(
        fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
        det_pix_size=0.196,
        n_slit=17,
        w_blur=spec_blur,
        pce=None,
        wavel_axis=instr_wavelength_axis,
        name="2A",
    )

    from surfh.DottestModels import R_Model

    wblurModel = R_Model.R_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)
    cube = wblurModel.mapsToCube(maps)

    assert dottest(wblurModel)
    

def test_RL_dottest():
    """
    Model : y = RLTx

    y : Hyperspectral slices of size (Nslices, L', Sx, Sy)
    R : Spectral blur operator
    L : Slicing operation
    x : Hyperspectral cube of size (4, Nx, Ny)
    """    
    templates = global_variable_testing.templates
    maps = global_variable_testing.maps
    instr_wavelength_axis = global_variable_testing.wavelength_axis
    n_lamnda = 200
    cube_walength_step = (instr_wavelength_axis[1]-instr_wavelength_axis[0])*2 # Set arbitrary wavel_axis 
    wavelength_axis = np.arange(instr_wavelength_axis[0], instr_wavelength_axis[-1], cube_walength_step)
    n_lamnda = len(wavelength_axis)
    sotf = global_variable_testing.sotf

    im_shape = global_variable_testing.im_shape
    local_shape = (im_shape[0]-100, im_shape[1]-100)

    cube = np.random.random((n_lamnda, im_shape[0]-150, im_shape[1]-150)) # Make smaller cube for easier memory computation
    out_cube = np.zeros_like(cube)

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


    grating_resolution = global_variable_testing.grating_resolution
    spec_blur = global_variable_testing.spec_blur

    # Def Channel spec.
    rchan = instru.IFU(
        fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
        det_pix_size=0.196,
        n_slit=17,
        w_blur=spec_blur,
        pce=None,
        wavel_axis=instr_wavelength_axis,
        name="2A",
    )

    from surfh.DottestModels import RL_Model

    rlModel = RL_Model.RL_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)

    assert dottest(rlModel, rtol=1e-2, echo=True)


def test_RLT_dottest():
    """
    Model : y = RLTx

    y : Hyperspectral slices of size (Nslices, L', Sx, Sy)
    R : Spectral blur operator
    L : Slicing operation
    x : Hyperspectral cube of size (4, Nx, Ny)
    """
    
    im_slice = slice(0,150,None)
    templates = global_variable_testing.templates
    maps = global_variable_testing.maps
    maps = maps[:,im_slice, im_slice]

    instr_wavelength_axis = global_variable_testing.chan_wavelength_axis
    wavelength_axis = global_variable_testing.wavelength_axis
    n_lamnda = len(wavelength_axis)
    sotf = global_variable_testing.sotf

    im_shape = global_variable_testing.im_shape
    im_shape = (im_slice.stop, im_slice.stop)

    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1])) # Make smaller cube for easier memory computation
    out_cube = np.zeros_like(cube)

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(cube.shape[1]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(cube.shape[2]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta


    grating_resolution = global_variable_testing.grating_resolution
    spec_blur = global_variable_testing.spec_blur

    # Def Channel spec.
    rchan = instru.IFU(
    fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur,
    pce=None,
    wavel_axis=instr_wavelength_axis,
    name="2A",
    )

    from surfh.DottestModels import RLT_Model

    rltModel = RLT_Model.RLT_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)

    assert dottest(rltModel, rtol=1e-3, echo=True)

def test_SigRLT_dottest():
    """
    Model : y = SigRLTx

    y : Hyperspectral slices of size (Nslices, L, Sx)
    Sig : Beta subsampling operator
    R : Spectral blur operator
    L : Slicing operator
    T : LMM operator
    x : Hyperspectral cube of size (4, Nx, Ny)
    """      
    im_slice = slice(0,150,None)
    templates = global_variable_testing.templates
    maps = global_variable_testing.maps
    maps = maps[:,im_slice, im_slice]

    instr_wavelength_axis = global_variable_testing.chan_wavelength_axis
    wavelength_axis = global_variable_testing.wavelength_axis
    n_lamnda = len(wavelength_axis)
    sotf = global_variable_testing.sotf

    im_shape = global_variable_testing.im_shape
    im_shape = (im_slice.stop, im_slice.stop)

    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1])) # Make smaller cube for easier memory computation
    out_cube = np.zeros_like(cube)

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(cube.shape[1]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(cube.shape[2]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta


    grating_resolution = global_variable_testing.grating_resolution
    spec_blur = global_variable_testing.spec_blur

    # Def Channel spec.
    rchan = instru.IFU(
    fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur,
    pce=None,
    wavel_axis=instr_wavelength_axis,
    name="2A",
    )

    from surfh.DottestModels import SigRLT_Model
    sigrltModel = SigRLT_Model.SigRLT_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)

    assert dottest(sigrltModel, rtol=1e-3, echo=True)
 
def test_SigRLCT_dottest():
    """
    Model : y = SigRLCTx

    y : Hyperspectral slices of size (Nslices, L, Sx)
    Sig : Beta subsampling operator
    R : Spectral blur operator
    L : Slicing operator
    C : Spatial convolution operator
    T : LMM operator
    x : Hyperspectral cube of size (4, Nx, Ny)
    """       
    import udft

    im_slice = slice(0,150,None)
    templates = global_variable_testing.templates
    maps = global_variable_testing.maps
    maps = maps[:,im_slice, im_slice]

    instr_wavelength_axis = global_variable_testing.chan_wavelength_axis
    wavelength_axis = global_variable_testing.wavelength_axis
    n_lamnda = len(wavelength_axis)

    spsf = global_variable_testing.spsf
    spsf = spsf[:,im_slice, im_slice]
    sotf = udft.ir2fr(spsf, (maps.shape[1], maps.shape[2]))

    im_shape = global_variable_testing.im_shape
    im_shape = (im_slice.stop, im_slice.stop)

    cube = np.random.random((n_lamnda, im_shape[0], im_shape[1])) # Make smaller cube for easier memory computation
    out_cube = np.zeros_like(cube)

    step = 0.025 # arcsec
    step_Angle = Angle(step, u.arcsec)

    cube_origin_alpha = 0
    cube_origin_beta = 0
    cube_alpha_axis = np.arange(cube.shape[1]).astype(np.float64)* step_Angle.degree
    cube_beta_axis = np.arange(cube.shape[2]).astype(np.float64)* step_Angle.degree
    cube_alpha_axis -= np.mean(cube_alpha_axis)
    cube_beta_axis -= np.mean(cube_beta_axis)
    cube_alpha_axis += cube_origin_alpha
    cube_beta_axis += cube_origin_beta


    grating_resolution = global_variable_testing.grating_resolution
    spec_blur = global_variable_testing.spec_blur

    # Def Channel spec.
    rchan = instru.IFU(
    fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur,
    pce=None,
    wavel_axis=instr_wavelength_axis,
    name="2A",
    )

    from surfh.DottestModels import SigRLCT_Model
    sigrlctModel = SigRLCT_Model.SigRLCT_spectro(sotf, templates, cube_alpha_axis, cube_beta_axis, wavelength_axis, rchan, step_Angle.degree)

    assert dottest(sigrlctModel, rtol=1e-3, echo=True)
 