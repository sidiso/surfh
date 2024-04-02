# SURFH - SUper Resolution and Fusion for Hyperspectral images
#
# Copyright (C) 2022 François Orieux <francois.orieux@universite-paris-saclay.fr>
# Copyright (C) 2018-2021 Ralph Abirizk <ralph.abirizk@universite-paris-saclay.fr>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

import aljabr
import numpy as np
import qmm
from numpy import ndarray as array

from surfh.Models import spectro, spectrolmm
from einops import einsum, rearrange


def vox_reconstruction(
    data: array,
    data_model: spectro.Spectro,
    spat_reg: float = 1.0,
    spat_th: float = 1.0,
    spec_reg: float = 1.0,
    spec_th: float = 1.0,
    init: array = None,
) -> array:
    spat_diff_r = aljabr.Diff(0, data_model.ishape)
    spat_diff_c = aljabr.Diff(1, data_model.ishape)
    spec_diff = aljabr.Diff(2, data_model.ishape)

    spat_prior_r = qmm.Objective(
        spat_diff_r.forward,
        spat_diff_r.adjoint,
        qmm.Huber(delta=spat_th),
        hyper=spat_reg,
        name="Row prior",
    )
    spat_prior_c = qmm.Objective(
        spat_diff_c.forward,
        spat_diff_c.adjoint,
        qmm.Huber(delta=spat_th),
        hyper=spat_reg,
        name="Col prior",
    )
    spec_prior = qmm.Objective(
        spec_diff.forward,
        spec_diff.adjoint,
        qmm.Huber(delta=spec_th),
        hyper=spec_reg,
        name="Spec prior",
    )

    data_adeq = qmm.QuadObjective(
        data_model.forward, data_model.adjoint, data=data, name="Data adeq"
    )

    if init is None:
        init = data_adeq.ht_data

    return qmm.mmmg(
        data_adeq + spat_prior_r + spat_prior_c + spec_prior, x0=init, max_iter=500
    )


def lmm_reconstruction(
    data: array,
    data_model: spectrolmm.SpectroLMM,
    spat_reg: float = 1.0,
    spat_th: float = 1.0,
    init: array = None,
) -> array:
    spat_diff_r = aljabr.Diff(0, data_model.ishape)
    spat_diff_c = aljabr.Diff(1, data_model.ishape)

    spat_prior_r = qmm.Objective(
        spat_diff_r.forward,
        spat_diff_r.adjoint,
        qmm.Huber(delta=spat_th),
        hyper=spat_reg,
        name="Row prior",
    )
    spat_prior_c = qmm.Objective(
        spat_diff_c.forward,
        spat_diff_c.adjoint,
        qmm.Huber(delta=spat_th),
        hyper=spat_reg,
        name="Col prior",
    )

    data_adeq = qmm.QuadObjective(
        data_model.forward, data_model.adjoint, data=data, name="Data adeq"
    )

    if init is None:
        init = data_adeq.ht_data

    return qmm.mmmg(data_adeq + spat_prior_r + spat_prior_c, x0=init, max_iter=500)

def partitioning_einops2(cube, di, dj):
    new_cube = rearrange(
        cube, "wl (dx bx) (dy by) -> wl (dx dy) bx by", dx=di, dy=dj
    )
    return new_cube




def concat_M(M):
    nb_blocks, _, nb_subblocks, _ = M.shape
    concat_width = nb_blocks * nb_subblocks
    concat = np.zeros((concat_width, concat_width), dtype=complex)  # always a square
    for l in range(nb_blocks):
        for c in range(nb_blocks):
            concat[
                l * nb_subblocks : (l + 1) * nb_subblocks,
                c * nb_subblocks : (c + 1) * nb_subblocks,
            ] += M[l, c, ...]
    return concat

# diff with concatenating: now works with decim different for both dimensions
def concatenating2(cubef, shape_target, di, dj):
    n_maps, d1_times_d2, h_block, w_block = cubef.shape
    h, w = shape_target
    
    concatenated_cube = np.zeros((n_maps, h, w), dtype=complex)
    k = 0
    for i in range(di):
        for j in range(dj):
            concatenated_cube[:, i * h_block : (i+1) * h_block, j * w_block : (j+1) * w_block] += cubef[:, k, :, :]
            k += 1
    
    return concatenated_cube


def split_M(M, split_shape):
    split = np.zeros(split_shape, dtype=complex)
    nb_blocks, _, nb_subblocks, _ = split_shape
    for l in range(nb_blocks):
        for c in range(nb_blocks):
            split[l, c, ...] += M[
                l * nb_subblocks : (l + 1) * nb_subblocks,
                c * nb_subblocks : (c + 1) * nb_subblocks,
            ]
    return split


def make_iHtH_spectro(HtH_freq_spectro):
    inv_hess_freq = np.zeros_like(HtH_freq_spectro, dtype=complex)
    H, W = inv_hess_freq.shape[-2:]
    for h in range(H):
        for w in range(W):
            M = np.copy(HtH_freq_spectro[..., h, w])
            C = concat_M(M)
            iC = np.linalg.inv(C)
            S = split_M(iC, inv_hess_freq.shape[:4])
            inv_hess_freq[..., h, w] += S
    return inv_hess_freq


# input in freq and not part, output in freq
def apply_hessian_freq(hess_spec_freq, di, dj, shape_target, x_freq):
    # partitionnement de x
    part_x_freq = partitioning_einops2(x_freq, di, dj)  # (5, 25, 50, 100)

    # produit de HtH avec x
    HtH_x_freq = hess_spec_freq * part_x_freq[np.newaxis, :, np.newaxis, :, :, :]
    # (5, 5, 25, 25, 50, 100) * (1, 5, 1, 25, 50, 100) = (5, 5, 25, 25, 50, 100)

    HtH_x_freq_sum = einsum(HtH_x_freq, "ti tj di dj h w -> ti di h w")

    # reconstitution des cartes en freq
    concat_HtH_x_freq = concatenating2(HtH_x_freq_sum, shape_target, di, dj)
    # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)

    return concat_HtH_x_freq