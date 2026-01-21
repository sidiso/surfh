#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:58:35 2023

@author: dpineau
"""

import numpy as np
from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir
from aljabr import Diff
import aljabr

# input x is not in freq, so we apply fft on x and then ifft
class Difference_Operator_Sep:  # separated gradients
    def __init__(self, shape_target):
        diff_kernel1 = diff_ir(2, 0)
        diff_kernel2 = diff_ir(2, 1)
        D1_freq = ir2fr(diff_kernel1, shape=shape_target, real=True)
        D2_freq = ir2fr(diff_kernel2, shape=shape_target, real=True)
        self.D1_freq = D1_freq
        self.D2_freq = D2_freq

    def D1(self, x):
        return irdftn(self.D1_freq[np.newaxis, ...] * rdft2(x), shape=x.shape[1:])

    def D1_t(self, x):
        return irdftn(
            np.conj(self.D1_freq[np.newaxis, ...]) * rdft2(x), shape=x.shape[1:]
        )

    def DtD1(self, x):
        return irdftn(
            np.abs(self.D1_freq[np.newaxis, ...]) ** 2 * rdft2(x), shape=x.shape[1:]
        )

    def D2(self, x):
        return irdftn(self.D2_freq[np.newaxis, ...] * rdft2(x), shape=x.shape[1:])

    def D2_t(self, x):
        return irdftn(
            np.conj(self.D2_freq[np.newaxis, ...]) * rdft2(x), shape=x.shape[1:]
        )

    def DtD2(self, x):
        return irdftn(
            np.abs(self.D2_freq[np.newaxis, ...]) ** 2 * rdft2(x), shape=x.shape[1:]
        )

# much faster than Difference_Operator_Sep, and supposed to give same output
class NpDiff_r(aljabr.LinOp): # dim = 3
    def __init__(self, maps_shape):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
    
    def forward(self, x):
        return - np.diff(np.pad(x, ((0, 0), (1, 0), (0, 0)), 'wrap'), axis=1)
    
    def adjoint(self, y):
        return np.diff(np.pad(y, ((0, 0), (0, 1), (0, 0)), 'wrap'), axis=1)

class NpDiff_c(aljabr.LinOp): # dim = 3
    def __init__(self, maps_shape):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
    
    def forward(self, x):
        return - np.diff(np.pad(x, ((0, 0), (0, 0), (1, 0)), 'wrap'), axis=2)
    
    def adjoint(self, y):
        return np.diff(np.pad(y, ((0, 0), (0, 0), (0, 1)), 'wrap'), axis=2)



# input x is in freq, and real=False, i.e. right side not troncated
# output is freq too
class Difference_Operator_Sep_Freq:  # separated gradients, input and output in freq
    def __init__(self, shape_target):
        # diff_kernel1 = diff_ir(2, 0)
        # diff_kernel2 = diff_ir(2, 1)
        
        diff_kernel1 = (np.array([-1, 1]))[..., np.newaxis]
        diff_kernel2 = (np.array([-1, 1]))[np.newaxis, ...]
        
        D1_freq = ir2fr(diff_kernel1, shape=shape_target, real=False)
        D2_freq = ir2fr(diff_kernel2, shape=shape_target, real=False)
        self.D1_freq = D1_freq
        self.D2_freq = D2_freq
        self.shape_target = shape_target

    def D1(self, x_freq):
        return self.D1_freq[np.newaxis, ...] * x_freq

    def D1_t(self, x_freq):
        return np.conj(self.D1_freq[np.newaxis, ...]) * x_freq

    def DtD1(self, x_freq):
        return (self.D1_freq[np.newaxis, ...] ** 2) * x_freq

    def D2(self, x_freq):
        return self.D2_freq[np.newaxis, ...] * x_freq

    def D2_t(self, x_freq):
        return np.conj(self.D2_freq[np.newaxis, ...]) * x_freq

    def DtD2(self, x_freq):
        return (self.D2_freq[np.newaxis, ...] ** 2) * x_freq
        

#%%

# garder celui là car modèle fusion fait avec laplacian

class Difference_Operator_Joint:  # gradients are joint
    def __init__(self, shape_target):
        diff_kernel = laplacian(2)
        D_freq = ir2fr(diff_kernel, shape=shape_target, real=True)
        self.D_freq = D_freq

    def D(self, x):
        return irdftn(self.D_freq[np.newaxis, ...] * rdft2(x), shape=x.shape[1:])

    def D_t(self, x):
        return irdftn(
            np.conj(self.D_freq[np.newaxis, ...]) * rdft2(x), shape=x.shape[1:]
        )

    def DtD(self, x):
        return irdftn(
            np.abs(self.D_freq[np.newaxis, ...]) ** 2 * rdft2(x), shape=x.shape[1:]
        )
