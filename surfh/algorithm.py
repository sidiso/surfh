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
        data_adeq + spat_prior_r + spat_prior_r + spec_prior, x0=init, max_iter=500
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

    return qmm.mmmg(data_adeq + spat_prior_r + spat_prior_r, x0=init, max_iter=500)
