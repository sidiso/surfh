import numpy as np
from jax import jit
import jax.numpy as jnp

from functools import partial


@jit # TODO is it useful here?
def dft(inarray): 
    return jnp.fft.rfftn(inarray, axes=range(-2, 0), norm="ortho")

@jit
def dft_mult(a, b):
    return jnp.fft.rfftn(a, axes=range(-2, 0), norm="ortho") * b


@partial(jit, static_argnums=1)
def idft(inarray, im_shape):
    return jnp.fft.irfftn(inarray, im_shape, axes=range(-len(im_shape), 0), norm="ortho")

@partial(jit, static_argnums=2)
def idft_mult(a, b, im_shape):
    c = jnp.multiply(a,b)
    return jnp.fft.irfftn(c, im_shape, axes=range(-len(im_shape), 0), norm="ortho")

@partial(jit, static_argnums=3)
def lmm_cube2maps_idft_mult(a, b, tpl, im_shape):
    c = jnp.multiply(a,b)
    d = idft(c, im_shape)
    return lmm_cube2maps(d, tpl)


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


#### wblur

@jit
def wblur(arr, wpsf):
    return jnp.sum(jnp.sum(
            # in [1, λ, α, β]
            jnp.expand_dims(arr, axis=0)
            # wpsf in [λ', λ, 1, β]
            * jnp.expand_dims(wpsf, axis=2),
            axis=1,
        ), axis=2)

@jit
def wblur_t(arr, wpsf):
    return jnp.sum(
        # in [λ', 1, α, β]
        jnp.expand_dims(arr, axis=1)
        # wpsf in [λ', λ, 1, β]
        * jnp.expand_dims(wpsf, axis=2),
        axis=0,
    )


### Others 
@jit
def fov_weight_jax(
    fov,
    slices,
    alpha_axis,
    beta_axis,
):
    """The weight windows of the FOV given slices of axis

    Notes
    -----
    Suppose the (floor, ceil) hypothesis of `LocalFOV.to_slices`.
    """
    alpha_step = alpha_axis[1] - alpha_axis[0]
    beta_step = beta_axis[1] - beta_axis[0]
    slice_alpha, slice_beta = slices

    selected_beta = beta_axis[slice_beta]

    weights = jnp.ones(
        (slice_alpha.stop - slice_alpha.start, slice_beta.stop - slice_beta.start)
    )

    # Weight for first α for all β
    # weights[0, :] *= (
    #     wght := abs(selected_alpha[0] - alpha_step / 2 - fov.alpha_start) / alpha_step
    # )
    # assert (
    #     0 <= wght <= 1
    # ), f"Weight of first alpha observed pixel in slit must be in [0, 1] ({wght:.2f})"

    if selected_beta[0] - beta_step / 2 < fov.beta_start:
        weights[:, 0] = (
            wght := 1
            - abs(selected_beta[0] - beta_step / 2 - fov.beta_start) / beta_step
        )
        assert (
            0 <= wght <= 1
        ), f"Weight of first beta observed pixel in slit must be in [0, 1] ({wght:.2f})"

    # weights[-1, :] *= (
    #     wght := abs(selected_alpha[-1] + alpha_step / 2 - fov.alpha_end) / alpha_step
    # )
    # assert (
    #     0 <= wght <= 1
    # ), f"Weight of last alpha observed pixel in slit must be in [0, 1] ({wght:.2f})"

    if selected_beta[-1] + beta_step / 2 > fov.beta_end:
        weights[:, -1] = (
            wght := 1
            - abs(selected_beta[-1] + beta_step / 2 - fov.beta_end) / beta_step
        )
        assert (
            0 <= wght <= 1
        ), f"Weight of last beta observed pixel in slit must be in [0, 1] ({wght:.2f})"

    return weights


