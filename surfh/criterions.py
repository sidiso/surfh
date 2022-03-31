import functools
import itertools as it

import numpy as np


#%% Hessian
#
def hessian_huber(
    forward_backward, spectral_output, mu_spat_idx, reg_col_otf, reg_row_otf, in_array
):
    """Compute A in Ax = b where A = H'H + mu_spat D'D + mu_spec D'D"""

    out = forward_backward(spectral_output, in_array)

    for idx, mu in enumerate(mu_spat_idx):
        out[idx, :, :] = out[idx, :, :] + mu * spatial_reg_hess(
            in_array[idx, :, :], reg_col_otf
        )
        out[idx, :, :] = out[idx, :, :] + mu * spatial_reg_hess(
            in_array[idx, :, :], reg_row_otf
        )
    return out


def hessian_huber_gr(
    forward_backward,
    spectral_output,
    mu_spat_idx,
    diffc_list,
    diffr_list,
    aux_col,
    aux_row,
    shape,
    in_array,
):
    """Compute A in Ax = b where A = H'H + mu_spat D'VD + mu_spec D'VD"""
    out = forward_backward(spectral_output, shape, in_array)
    for idx, mu in enumerate(mu_spat_idx):
        out = out + mu * diffc_list[idx].adjoint(
            aux_col[idx, ...] * diffc_list[idx](in_array)
        )
        out = out + mu * diffr_list[idx].adjoint(
            aux_row[idx, ...] * diffr_list[idx](in_array)
        )
    return out


def spatial_reg_hess(in_array, reg_otf):
    return np.abs(reg_otf[np.newaxis, :, :]) ** 2 * in_array


def huber_spat_aux(in_f, huber, col_reg_otf, row_reg_otf, shape):
    diff_col = idft(spatial_reg(in_f, col_reg_otf), shape)
    diff_row = idft(spatial_reg(in_f, row_reg_otf), shape)

    return dft(huber.min_gy(diff_col)), dft(huber.min_gy(diff_row))


def huber_spat_aux_gr(in_array, potc, potr, diffc, diffr):
    diff_col = potc.gr_coeffs(diffc(in_array))
    diff_row = potr.gr_coeffs(diffr(in_array))
    return diff_col, diff_row


#
def reg_transpose(aux_col, aux_row, col_reg_otf, row_reg_otf, mu_col, mu_row):
    out = mu_col * spatial_reg_transpose(aux_col, col_reg_otf)
    out = out + mu_row * spatial_reg_transpose(aux_row, row_reg_otf)
    return out


def spatial_reg(in_array_f, reg_otf):
    return reg_otf[np.newaxis, :, :] * in_array_f


def spatial_reg_transpose(in_f, reg_otf):
    return reg_otf.conj()[np.newaxis, :, :] * in_f


#%% Preconditioner


def precond_setup(
    wavel_axis, alpha_step, shape, components, col_reg_otf, row_reg_otf, mu_spat
):
    H_otf = models_LMM.precalc_otf(wavel_axis, alpha_step, shape, components)
    H_otf = H_otf / H_otf.shape[1]
    out = (
        np.sum(np.abs(H_otf) ** 2, axis=1)
        + mu_spat * np.abs(col_reg_otf[np.newaxis, :, :]) ** 2
        + mu_spat * np.abs(row_reg_otf[np.newaxis, :, :]) ** 2
    )
    return out


def precond_MDFT_setup(
    wavel_axis, alpha_step, shape, components, col_reg_otf, row_reg_otf, mu_spat
):
    H_otf = models_LMM.precalc_otf(wavel_axis, alpha_step, shape, components)

    HTH_otf = np.zeros(
        (H_otf.shape[0], H_otf.shape[0], H_otf.shape[2], H_otf.shape[3]),
        dtype="complex",
    )
    for i, j in it.product(range(H_otf.shape[0]), range(H_otf.shape[0])):
        HTH_otf[i, j, ...] = np.sum(np.conj(H_otf[i, ...]) * H_otf[j, ...], axis=0)

    for i in range(components.shape[0]):
        HTH_otf[i, i::] += (
            mu_spat * np.abs(col_reg_otf) ** 2 + mu_spat * np.abs(row_reg_otf) ** 2
        )
    inv_A_f = np.zeros_like(HTH_otf)
    for i, j in it.product(range(HTH_otf.shape[2]), range(HTH_otf.shape[3])):
        inv_A_f[:, :, i, j] = np.linalg.inv(HTH_otf[:, :, i, j])

    return inv_A_f
