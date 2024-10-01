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

from . import models


def vox_reconstruction(
    data: array,
    data_model: models.Spectro,
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
    data_model: models.SpectroLMM,
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
