import numpy as np
from jax import jit
import jax.numpy as jnp

from functools import partial


@jit # TODO is it useful here?
def dft(inarray): 
    return jnp.fft.rfftn(inarray, axes=range(-2, 0), norm="ortho")

@partial(jit, static_argnums=1)
def idft(inarray, im_shape):
    return jnp.fft.irfftn(inarray, im_shape, axes=range(-len(im_shape), 0), norm="ortho")

@jit
def lmm_maps2cube(maps, tpls):
    cube = jnp.sum(
            jnp.expand_dims(maps, 1) * tpls[..., jnp.newaxis, jnp.newaxis], axis=0
        )
    return cube

@jit
def lmm_cube2maps(cube, tpls):
    maps = jnp.concatenate(
            [
                jnp.sum(cube * tpl[..., jnp.newaxis, jnp.newaxis], axis=0)[jnp.newaxis, ...]
                for tpl in tpls
            ],
            axis=0,
            )
    return maps