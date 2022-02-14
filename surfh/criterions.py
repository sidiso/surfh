import numpy as np
from mrs_functions import models_LMM
from mrs_functions import udft
import itertools as it

# import pywt
from edwin import optim
from edwin.tools import Timer
import functools


def dft(in_array):
    return udft.rfftn(in_array, axes=(1, 2))


def idft(in_array, shape):
    return udft.irfftn(in_array, axes=(1, 2), s=shape[:2])


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


#%%
def precondit(H_otf, shape, in_array):
    in_f = dft(in_array)
    return udft.irfftn(np.multiply(in_f, 1 / H_otf), axes=(1, 2), s=shape)


def precondit_MDFT(inv_HTH_otf, in_f):
    return np.sum(inv_HTH_otf * in_f[np.newaxis, :, :, :], axis=1)


#%%


def reconst_quad(res, hessian, shape, data_transpose, setup=None, precondit=None):

    with Timer() as t:

        res, info = optim.conj_grad(hessian, res, data_transpose, setup, precondit)
    print("CG : ", t.secs)
    res_quad = udft.irfftn(res, axes=(1, 2), s=shape)
    return res_quad, info


# -------------------------------------------------------------------------------------------


def reconst_GY(
    res_gy,
    hessian,
    huber_spatial,
    shape,
    mu_spat_idx,
    data_transpose,
    col_reg_otf,
    row_reg_otf,
    f_crit,
    n_iter,
    setup=None,
    precondit=None,
):

    aux_col = np.zeros_like(data_transpose)
    aux_row = np.zeros_like(data_transpose)
    criterion = []
    with Timer() as t:
        for iteration in range(n_iter):

            tmp = np.zeros_like(data_transpose)
            for idx, i in enumerate(mu_spat_idx):
                tmp[idx, :, :] = reg_transpose(
                    aux_col[idx, :, :],
                    aux_row[idx, :, :],
                    col_reg_otf,
                    row_reg_otf,
                    mu_spat_idx[idx],
                    mu_spat_idx[idx],
                )  # Compute D^t b for only for the chosen maps(s)
            second_term = data_transpose + tmp

            res_gy, info = optim.conj_grad(
                hessian, res_gy, second_term, setup, precondit
            )

            criterion.extend(info["crit_val"])

            for idx, i in enumerate(mu_spat_idx):
                    aux_col[idx, :, :], aux_row[idx, :, :] = huber_spat_aux(
                        res_gy[idx, :, :],
                        huber_spatial,
                        col_reg_otf,
                        row_reg_otf,
                        shape,
                    )  # update the auxilary variables only for the chosen map(s)

        res_huber_gy = udft.irfftn(res_gy, axes=(1, 2), s=shape)
        res_col = udft.irfftn(aux_col, axes=(1, 2), s=shape)
        res_row = udft.irfftn(aux_row, axes=(1, 2), s=shape)
    print("GY:", t.secs)
    return res_huber_gy, res_col, res_row, info, criterion


# -------------------------------------------------------------------------------------------


def reconst_GR(
    res_gr,
    potc_list,
    potr_list,
    shape,
    mu_spat_idx,
    data_transpose,
    diffc_list,
    diffr_list,
    ifu,
    spectral_output,
    f_crit,
    n_iter,
    setup=None,
    precondit=None,
):

    aux_col = np.ones((data_transpose.shape[0],) + shape)
    aux_row = np.ones((data_transpose.shape[0],) + shape)
    # crit_value= [] 
    # criter_time = []
    norm_grad = [] 
    with Timer() as t:
        for iteration in range(n_iter):
            hessian = functools.partial(
                hessian_huber_gr,
                ifu.forward_backward,
                spectral_output,
                mu_spat_idx,
                diffc_list,
                diffr_list,
                aux_col,
                aux_row,
                shape,
            )
            res_gr, info = optim.conj_grad(
                hessian, res_gr, data_transpose, setup, precondit
            )
            ## If we want to compute the crit val, norm_grad... use the script optimplus instead of optim
            #    crit_value.extend(info["crit_val"])
            #  criter_time.extend(info["criter_time"])
            #  if crit_value[-1] <=  setup["cg threshold"][1] or norm_grad[-1] <=  setup["cg threshold"][0] :
            #      break
            
            
            norm_grad.extend(info["norm_grad"])
            if norm_grad[-1] <setup["cg threshold"]:
                break

            
            for idx, i in enumerate(mu_spat_idx):

                aux_col[idx, :, :], aux_row[idx, :, :] = huber_spat_aux_gr(
                    res_gr,
                    potc_list[idx],
                    potr_list[idx],
                    diffc_list[idx],
                    diffr_list[idx],
                )

    # res_huber_gr = udft.irfftn(res_gr, axes=(1, 2), s=shape)
    print("GR:", t.secs)
    return res_gr, aux_col, aux_row, info, norm_grad


# TV Chambolle----------------------------------------------------------------------------------------------------------------
def chambolle_pock(P, P_t, data, Lambda, L, n_it, return_energy=True, verbose=False):
    """
    Chambolle-Pock algorithm for the minimization of the objective function
        ||P*x - d||_2^2 + Lambda*TV(x)

    P : projection operator (or H)
    PT : backprojection operator (or H_t)
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    """
    # Set parameter
    sigma = 1.0 / L
    tau = 1.0 / L
    theta = 1.0

    x = 0 * P_t(data)
    p = 0 * gradient(x)
    q = 0 * data
    x_tilde = 0 * x
    # def hessian_huber(
    #    forward_backward, spectral_output,shape, mu_spat, reg_col_otf, reg_row_otf, in_array
    # ):
    #    """Compute A in Ax = b where A = H'H + mu_spat D'D + mu_spec D'D"""
    #
    #    out = forward_backward(spectral_output, in_array)
    #
    #    out += mu_spat * spatial_reg_hess(in_array, reg_col_otf)
    #    out += mu_spat * spatial_reg_hess(in_array, reg_row_otf)
    #    return out

    if return_energy:
        en = np.zeros(n_it)

    for k in range(0, n_it):

        # Update dual variables
        p = proj_l2(p + sigma * gradient(x_tilde), Lambda)
        q = (q + sigma * P(x_tilde) - sigma * data) / (1.0 + sigma)

        # Update primal variables
        x_old = x
        x = x + tau * div(p) - tau * P_t(q)  # eq 2 of Algo 3
        x_tilde = x + theta * (x - x_old)  # eq 3 of Algo 3

        # Calculate norms
        if return_energy:
            fidelity = 0.5 * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = 1.0 * fidelity + Lambda * tv
            en[k] = energy
            if verbose and k % 20 == 0:
                print(
                    "[%d] : energy %2.2e \t fidelity %2.2e \t TV %2.2e"
                    % (k, energy, fidelity, tv)
                )
    if return_energy:
        return en, x
    else:
        return x


def power_method(P, PT, data, n_it=10):
    """
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    """
    x = PT(data)
    for k in range(0, n_it):
        x = PT(P(x)) - div(gradient(x))
        s = np.sqrt(norm2sq(x))
        x /= s
    return np.sqrt(s)


def div(grad):
    """
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def gradient(img):
    """
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    shape = [img.ndim,] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [
        0,
        slice(None, -1),
    ]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())


def norm1(mat):
    return np.sum(np.abs(mat))


def proj_l2(g, Lambda=1.0):
    """
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    """
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g ** 2, 0)) / Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res


# --------------------------------------------------------------------------------
