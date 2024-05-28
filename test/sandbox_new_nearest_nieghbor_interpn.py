import pytest
import numpy as np
from aljabr import LinOp, dottest
import global_variable_testing
import matplotlib.pyplot as plt

import scipy as sp
from scipy import misc
from surfh.ToolsDir import jax_utils, python_utils, cython_utils, utils
from surfh.Models import instru
from astropy import units as u
from astropy.coordinates import Angle
from numpy.random import standard_normal as randn
from scipy.spatial import cKDTree


def griddata(points, values, xi, method='linear', fill_value=np.nan,
             rescale=False):

    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    ip = NearestNDInterpolator(points, values, rescale=rescale)
    return ip(xi)


def _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.

    """
    j = n = 0

    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        points = np.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[...,j] = item
    else:
        points = np.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points

class NDInterpolatorBase:
    def __init__(self, points, values, fill_value=np.nan, ndim=None,
                 rescale=False, need_contiguous=True, need_values=True):
        """
        Check shape of points and values arrays, and reshape values to
        (npoints, nvalues).  Ensure the `points` and values arrays are
        C-contiguous, and of correct type.
        """

        self.tri = None

        points = _ndim_coords_from_arrays(points)

        if need_contiguous:
            points = np.ascontiguousarray(points, dtype=np.float64)

        if not rescale:
            self.scale = None
            self.points = points
        else:
            # scale to unit cube centered at 0
            self.offset = np.mean(points, axis=0)
            self.points = points - self.offset
            self.scale = np.ptp(points, axis=0)
            self.scale[~(self.scale > 0)] = 1.0  # avoid division by 0
            self.points /= self.scale
        
        self._calculate_triangulation(self.points)
        
        if need_values or values is not None:
            self._set_values(values, fill_value, need_contiguous, ndim)
        else:
            self.values = None

    def _calculate_triangulation(self, points):
        pass

    def _set_values(self, values, fill_value=np.nan, need_contiguous=True, ndim=None):
        values = np.asarray(values)
        _check_init_shape(self.points, values, ndim=ndim)

        self.values_shape = values.shape[1:]
        if values.ndim == 1:
            self.values = values[:,None]
        elif values.ndim == 2:
            self.values = values
        else:
            self.values = values.reshape(values.shape[0],
                                            np.prod(values.shape[1:]))
        
        # Complex or real?
        self.is_complex = np.issubdtype(self.values.dtype, np.complexfloating)
        if self.is_complex:
            if need_contiguous:
                self.values = np.ascontiguousarray(self.values,
                                                    dtype=np.complex128)
            self.fill_value = complex(fill_value)
        else:
            if need_contiguous:
                self.values = np.ascontiguousarray(
                    self.values, dtype=np.float64
                )
            self.fill_value = float(fill_value)

    def _check_call_shape(self, xi):
        xi = np.asanyarray(xi)
        if xi.shape[-1] != self.points.shape[1]:
            raise ValueError("number of dimensions in xi does not match x")
        return xi

    def _scale_x(self, xi):
        if self.scale is None:
            return xi
        else:
            return (xi - self.offset) / self.scale

    def _preprocess_xi(self, *args):
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        interpolation_points_shape = xi.shape
        xi = xi.reshape(-1, xi.shape[-1])
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        return self._scale_x(xi), interpolation_points_shape


    def __call__(self, *args):
        """
        interpolator(xi)

        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn: array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        """
        xi, interpolation_points_shape = self._preprocess_xi(*args)

        if self.is_complex:
            r = self._evaluate_complex(xi)
        else:
            r = self._evaluate_double(xi)

        return np.asarray(r).reshape(interpolation_points_shape[:-1] + self.values_shape)


def _check_init_shape(points, values, ndim=None):
    """
    Check shape of points and values arrays

    """
    if values.shape[0] != points.shape[0]:
        raise ValueError("different number of values and points")
    if points.ndim != 2:
        raise ValueError("invalid shape for input data points")
    if points.shape[1] < 2:
        raise ValueError("input data must be at least 2-D")
    if ndim is not None and points.shape[1] != ndim:
        raise ValueError("this mode of interpolation available only for "
                         "%d-D data" % ndim)

class NearestNDInterpolator(NDInterpolatorBase):
    def __init__(self, x, y, rescale=False, tree_options=None):
        NDInterpolatorBase.__init__(self, x, y, rescale=rescale,
                                    need_contiguous=False,
                                    need_values=False)
        if tree_options is None:
            tree_options = dict()
        self.tree = cKDTree(self.points, **tree_options)
        self.values = np.asarray(y)

    def __call__(self, *args, **query_options):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        **query_options
            This allows ``eps``, ``p``, ``distance_upper_bound``, and ``workers``
            being passed to the cKDTree's query function to be explicitly set.
            See `scipy.spatial.cKDTree.query` for an overview of the different options.

            .. versionadded:: 1.12.0

        """
        # For the sake of enabling subclassing, NDInterpolatorBase._set_xi performs
        # some operations which are not required by NearestNDInterpolator.__call__, 
        # hence here we operate on xi directly, without calling a parent class function.
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)

        # We need to handle two important cases:
        # (1) the case where xi has trailing dimensions (..., ndim), and
        # (2) the case where y has trailing dimensions
        # We will first flatten xi to deal with case (1),
        # do the computation in flattened array while retaining y's dimensionality,
        # and then reshape the interpolated values back to match xi's shape.

        # Flatten xi for the query
        xi_flat = xi.reshape(-1, xi.shape[-1])
        original_shape = xi.shape
        flattened_shape = xi_flat.shape

        # if distance_upper_bound is set to not be infinite,
        # then we need to consider the case where cKDtree
        # does not find any points within distance_upper_bound to return.
        # It marks those points as having infinte distance, which is what will be used
        # below to mask the array and return only the points that were deemed
        # to have a close enough neighbor to return something useful.
        dist, i = self.tree.query(xi_flat, **query_options)
        valid_mask = np.isfinite(dist)

        # create a holder interp_values array and fill with nans.
        if self.values.ndim > 1:
            interp_shape = flattened_shape[:-1] + self.values.shape[1:]
        else:
            interp_shape = flattened_shape[:-1]

        if np.issubdtype(self.values.dtype, np.complexfloating):
            interp_values = np.full(interp_shape, np.nan, dtype=self.values.dtype)
        else:
            interp_values = np.full(interp_shape, np.nan)

        interp_values[valid_mask] = self.values[i[valid_mask], ...]

        if self.values.ndim > 1:
            new_shape = original_shape[:-1] + self.values.shape[1:]
        else:
            new_shape = original_shape[:-1]
        interp_values = interp_values.reshape(new_shape)

        return interp_values

