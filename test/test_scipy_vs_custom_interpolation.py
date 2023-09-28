import itertools as it
import sys
from functools import partial
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import udft
import time
from astropy.io import fits
from scipy.signal import convolve as conv
from scipy.signal import convolve2d as conv2

from surfh import instru, models
from surfh import smallmiri as miri
from surfh import utils

import scipy_python_interpolate 
import scipy_optimize_python_interpolate
import scipy_optimize_cython_interpolate


def orion():
    """Rerturn maps, templates, spatial step and wavelength"""
    maps = fits.open("./cube_orion/abundances_orion.fits")[0].data

    h2_map = maps[0]
    if_map = maps[1]
    df_map = maps[2]
    mc_map = maps[3]

    spectrums = fits.open("./cube_orion/spectra_mir_orion.fits")[1].data
    wavel_axis = spectrums.wavelength

    h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
    if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
    df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
    mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

    return (
        np.asarray((h2_map, if_map, df_map, mc_map)),
        np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
        0.025,
        wavel_axis,
    )


def dft(inarray):
    """Apply the unitary Discret Fourier Transform on last two axis.

    Parameters
    ----------
    inarray: array-like
      The array to transform

    Notes
    -----
    Use `scipy.fft.rfftn` with `workers=-1`.
    """
    return sp.fft.rfftn(inarray, axes=range(-2, 0), norm="ortho", workers=-1)


maps, tpl, step, wavel_axis = orion()
spat_ss = 4
ir = np.ones((spat_ss, spat_ss)) / spat_ss ** 2
maps = np.asarray([conv2(arr, ir)[::spat_ss, ::spat_ss] for arr in maps])
step = instru.get_step([chan.det_pix_size for chan in miri.all_chan], 5)
srfs = instru.get_srf(
    [chan.det_pix_size for chan in miri.all_chan],
    step,
)
alpha_axis = np.arange(maps.shape[1]) * step
beta_axis = np.arange(maps.shape[2]) * step
alpha_axis -= np.mean(alpha_axis)
beta_axis -= np.mean(beta_axis)
alpha_axis += +miri.ch1a.fov.origin.alpha
beta_axis += +miri.ch1a.fov.origin.beta

tpl_ss = 3
ir = np.ones((1, tpl_ss)) / tpl_ss
tpl = conv2(tpl, ir, "same")[:, ::tpl_ss]
wavel_axis = wavel_axis[::tpl_ss]

spsf = utils.gaussian_psf(wavel_axis, step)

if "cube" not in globals():
    print("Compute cube")
    cube = np.sum(np.expand_dims(maps, 1) * tpl[..., np.newaxis, np.newaxis], axis=0)
if "sotf" not in globals():
    print("Compute SPSF")
    sotf = udft.ir2fr(spsf, maps.shape[1:])

#%% Models
from importlib import resources
with resources.path("surfh.data", "mrs_recommended_dither.dat") as path:
    dithering = np.loadtxt(path, delimiter=",")
ch1_dither = instru.CoordList.from_array(dithering[:1, :])

main_pointing = instru.Coord(0, 0)
pointings = instru.CoordList(c + main_pointing for c in ch1_dither).pix(step)
# pointings = instru.CoordList([ifu.Coord(5 * step, 5 * step)])

spectro = models.Spectro(
    [miri.ch1a],
    alpha_axis,
    beta_axis,
    wavel_axis,
    sotf,
    pointings,
)



out = np.zeros(spectro.oshape)
print("Spectro oshape is ", spectro.oshape)
blurred_f = dft(cube)* spectro.sotf

_idx = np.cumsum([0] + [np.prod(chan.oshape) for chan in spectro.channels])
oshape = (_idx[-1],)
data = np.zeros(oshape)

# fORWARD
for idx, chan in enumerate(spectro.channels):
    #out[self._idx[idx] : self._idx[idx + 1]] = chan.forward(blurred_f).ravel()

    blurred = chan.sblur(blurred_f[chan.wslice, ...])
    start = time.time()
    for p_idx, pointing in enumerate(chan.pointings):           
        gridded = chan.gridding(blurred, pointing)

        alpha_coord, beta_coord = (chan.instr.fov + pointing).local2global(
            chan.local_alpha_axis, chan.local_beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(blurred.shape[0])

        out_shape = (len(wl_idx),) + alpha_coord.shape

        local_coords = np.vstack(
            [
                np.repeat(
                    np.repeat(wl_idx.reshape((-1, 1, 1)), out_shape[1], axis=1),
                    out_shape[2],
                    axis=2,
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], out_shape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], out_shape[0], axis=0).ravel(),
            ]
        ).T

        # This output can be processed in local ref.
        start1 = time.time()
        scipy_result= sp.interpolate.interpn( (wl_idx, chan.alpha_axis, chan.beta_axis), blurred, local_coords).reshape(out_shape)
        end1 = time.time()
        start2 = time.time()
        #python_result = scipy_python_interpolate.interpn( (wl_idx, chan.alpha_axis, chan.beta_axis), blurred, local_coords).reshape(out_shape)
        end2 = time.time()
        start3 = time.time()
        nWave = local_coords[-1,0] + 1
        custom_local_coord =(local_coords[0:int(local_coords.shape[0]//nWave),:])[:,1:3]
        custom_result = scipy_optimize_python_interpolate.interpn( (chan.alpha_axis, chan.beta_axis), blurred, custom_local_coord, local_coords.shape, nWave).reshape(out_shape)
        end3 = time.time()
        start4 = time.time()
        cython_result = scipy_optimize_cython_interpolate.interpn( (chan.alpha_axis, chan.beta_axis), blurred, custom_local_coord, local_coords.shape, nWave).reshape(out_shape)
        end4 = time.time()


print("Scipy time is", end1-start1)
print("Scipy python time is", end2-start2)
print("Custom Scipy python time is", end3-start3)
print("Custom Cython time is", end4-start4)
#print("is Scipy and Python Scipy the same ? ", np.allclose(scipy_result, python_result))
print("is Scipy and Custom python the same ? ", np.allclose(scipy_result, custom_result))
print("is Scipy and Cython python the same ? ", np.allclose(scipy_result, cython_result))


""" data = spectro.forward(cube)

print("#################################\n\n")
print("#################################")
# Adjoint
tmp = np.zeros(
    spectro.ishape[:2] + (spectro.ishape[2] // 2 + 1,), dtype=np.complex128
)
for idx, chan in enumerate(spectro.channels):
    out = np.zeros(chan.ishape, dtype=np.complex128)
    scipy_blurred = np.zeros(chan.cshape)
    python_blurred = np.zeros(chan.cshape)
    custom_blurred = np.zeros(chan.cshape)
    for p_idx, pointing in enumerate(chan.pointings):
        gridded = np.zeros(chan.local_shape)
        for slit_idx in range(chan.instr.n_slit):
            sliced = np.zeros(chan.slit_shape(slit_idx))

            sliced[:, : chan.oshape[3] * chan.srf : chan.srf] = chan.wblur_t(
                np.repeat(
                    np.expand_dims(
                        np.reshape(data[spectro._idx[idx] : spectro._idx[idx + 1]], chan.oshape) [p_idx, slit_idx] * chan.instr.pce[..., np.newaxis],
                        axis=2,
                    ),
                    sliced.shape[2],
                    axis=2,
                )
            )
            gridded += chan.slicing_t(sliced, slit_idx)

        alpha_coord, beta_coord = (chan.instr.fov + pointing).global2local(
            chan.alpha_axis, chan.beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(gridded.shape[0])

        out_shape = (len(wl_idx), len(chan.alpha_axis), len(chan.beta_axis))

        global_coords = np.vstack(
            [
                np.tile(
                    wl_idx.reshape((-1, 1, 1)), (1, out_shape[1], out_shape[2])
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], out_shape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], out_shape[0], axis=0).ravel(),
            ]
        ).T

        # This output can be processed in local ref.
        startT1 = time.time()
        scipy_blurred += sp.interpolate.interpn(
            (wl_idx, chan.local_alpha_axis, chan.local_beta_axis),
            gridded,
            global_coords,
            bounds_error=False,
            fill_value=0,
        ).reshape(out_shape)
        endT1 = time.time()

        startT2 = time.time()
        python_blurred += scipy_python_interpolate.interpn((wl_idx, chan.local_alpha_axis, chan.local_beta_axis),
            gridded,
            global_coords,
            bounds_error=False,
            fill_value=0,
        ).reshape(out_shape)
        endT2 = time.time()

        startT3 = time.time()
        nWave = global_coords[-1,0] + 1
        custom_global_coord =(global_coords[0:int(global_coords.shape[0]//nWave),:])[:,1:3]

        custom_blurred += scipy_optimize_python_interpolate.interpn( (chan.local_alpha_axis, chan.local_beta_axis), 
                                                                    gridded, 
                                                                    custom_global_coord, 
                                                                    global_coords.shape, 
                                                                    nWave, 
                                                                    bounds_error=False, 
                                                                    fill_value=0,).reshape(out_shape)
        endT3 = time.time()


print("Scipy Transpose Interpolation time is ", endT1 - startT1)
print("Python Transpose Interpolation time is ", endT2 - startT2)
print("Custom Transpose Interpolation time is ", endT3 - startT3)

print("is Scipy.T and Python Scipy.T the same ? ", np.allclose(scipy_blurred, python_blurred))
print("is Scipy.T and Custom python.T the same ? ", np.allclose(scipy_blurred, custom_blurred)) """






