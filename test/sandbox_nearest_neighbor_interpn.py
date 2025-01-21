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
from surfh.ToolsDir import nearest_neighbor_interpolation


def _check_points(points):
    descending_dimensions = []
    grid = []
    # i is the dimension, p all the coordinates of this dimension
    for i, p in enumerate(points):
        # early make points float
        # see https://github.com/scipy/scipy/pull/17230
        p = np.asarray(p, dtype=float)
        if not np.all(p[1:] > p[:-1]):
            if np.all(p[1:] < p[:-1]):
                # input is descending, so make it ascending
                descending_dimensions.append(i)
                p = np.flip(p)
            else:
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending or descending" % i)
        # see https://github.com/scipy/scipy/issues/17716
        p = np.ascontiguousarray(p)
        grid.append(p)
    return tuple(grid), tuple(descending_dimensions)

def find_interval_ascending(x,
                            nx,
                            xval,
                            prev_interval=0,
                            extrapolate=1):
    """
    Find an interval such that x[interval] <= xval < x[interval+1]. Assuming
    that x is sorted in the ascending order.
    If xval < x[0], then interval = 0, if xval > x[-1] then interval = n - 2.

    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.

    """
    # high, low, mid
    high = low = mid = 0
    interval = prev_interval
    a = x[0]
    b = x[nx - 1]
    if interval < 0 or interval >= nx:
        interval = 0

    # Check if the value to find is in the range of the input value
    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    # Check if the value to find is at eh boundery
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)

        # Split the value to search in half (not equal )
        if xval >= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        # En gros c'est un "Guess the number", en prenant la moitié de l'intervale entre high et low à chaque iteration
        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval


def find_indices(grid, xi):

    # Axes to iterate over
    I = xi.shape[0] # = 2 because we took the Transpose of xi
    J = xi.shape[1] # = 66203 because we took the Transpose of xi
    index = 0
    # Indices of relevant edges between which xi are situated
    indices = np.empty_like(xi, dtype=int)
    # Distances to lower edge in unity units
    norm_distances = np.zeros_like(xi, dtype=float)

    # iterate through dimensions
    for i in range(I):
        
        grid_i = grid[i] # = array of 512 element
        grid_i_size = grid_i.shape[0] # = 512, the size of the input grid

        for j in range(J):
            value = xi[i, j] # Select one one of the 2D coordinate of xi

            # Return the low idx of the input grid corresponding to the current output Alpha or Beta coordinate
            index = find_interval_ascending(grid_i,
                                            grid_i_size,
                                            value,
                                            prev_interval=index,
                                            extrapolate=1)
            indices[i, j] = index


            # Return the equivalent of "step", the Alpha or Beta distance between 2 input pixels grid
            denom = grid_i[index + 1] - grid_i[index]

            # Compute the distance
            norm_distances[i, j] = (value - grid_i[index]) / denom


    return np.asarray(indices), np.asarray(norm_distances)


def solve_2D_hypercube(c_indices, c_norm_distances, 
                       c_in_values, shape_j, nWave):

    num_points = shape_j
    j = 0
    i0 = i1 = 0
    y0 = y1 = 0.
    w1 = w2 = w3 = w4 = 0.
    c_out_values = np.ascontiguousarray(np.zeros((nWave, num_points)))
    nearest_coord = np.zeros((num_points, 2))

    for j in range(num_points):
        i0 = c_indices[0, j]
        i1 = c_indices[1, j]

        y0 = c_norm_distances[0,j]
        y1 = c_norm_distances[1,j]

        w1 = ((1. - y0) * (1. - y1))
        w2 = ((1. - y0) * y1)
        w3 = (y0 * (1. - y1))
        w4 = (y0 * y1)


        min_idx = np.argmin([w1, w2, w3, w4])

        nearest_coord[j,0] = i0 + min_idx%2
        nearest_coord[j,1] = i1 + int(min_idx>=2)

        # for l in range(nWave):
        #     c_out_values[l,j] = c_in_values[l, i0, i1]*w1
        #     c_out_values[l,j] = c_out_values[l,j] + c_in_values[l, i0, i1+1]*w2
        #     c_out_values[l,j] = c_out_values[l,j] + c_in_values[l, i0+1, i1]*w3
        #     c_out_values[l,j] = c_out_values[l,j] + c_in_values[l, i0+1, i1+1]*w4

    return np.asarray(nearest_coord)



class RegularGridInterpolator:

    _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}
    _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
    _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS

    def __init__(self, points, values, nWave, method="linear", bounds_error=True,
                 fill_value=np.nan):
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)

        self.method = method
        self.bounds_error = bounds_error

        # Check the points
        # Grid is the new name of points and (xa, xb)
        # descending_dimention check is value of coordinates are increasing or decreasing grosso modo
        self.grid, self._descending_dimensions = _check_points(points)

        #Set value of "im" to float
        self.values = self._check_values(values)
        self.nWave = int(nWave)

        # Set fill value to np.nan
        self.fill_value = self._check_fill_value(self.values, fill_value)
        if self._descending_dimensions:
            self.values = np.flip(values, axis=self._descending_dimensions)


    def _check_points(self, points):
        return _check_points(points)

    def _check_values(self, values):
        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        return values

    def _check_fill_value(self, values, fill_value):
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
        return fill_value


    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to evaluate the interpolator at.

        method : str, optional
            The method of interpolation to perform. Supported are "linear",
            "nearest", "slinear", "cubic", "quintic" and "pchip". Default is
            the method chosen when the interpolator was created.

        Returns
        -------
        values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at `xi`. See notes for behaviour when
            ``xi.ndim == 1``.

        Notes
        -----
        In the case that ``xi.ndim == 1`` a new axis is inserted into
        the 0 position of the returned array, values_x, so its shape is
        instead ``(1,) + values.shape[ndim:]``.

        Examples
        --------
        Here we define a nearest-neighbor interpolator of a simple function

        By construction, the interpolator uses the nearest-neighbor
        interpolation

        """
        method = self.method if method is None else method
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)

        xi, xi_shape, ndim, nans, out_of_bounds = self._prepare_xi(xi)

        indices, norm_distances = find_indices(self.grid, xi.T)

        result = solve_2D_hypercube(indices, norm_distances, self.values, indices.shape[1], self.nWave)
        


        # if not self.bounds_error and self.fill_value is not None:
        #     result[:,out_of_bounds] = self.fill_value

        # # f(nan) = nan, if any
        # if np.any(nans):
        #     result[nans] = np.nan
        return result


    def _prepare_xi(self, xi):
        ndim = len(self.grid)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             f"{xi.shape[-1]} but this "
                             f"RegularGridInterpolator has dimension {ndim}")

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = np.asarray(xi, dtype=float)

        # find nans in input
        nans = np.any(np.isnan(xi), axis=-1)

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)
            out_of_bounds = None
        else:
            out_of_bounds = self._find_out_of_bounds(xi.T)

        return xi, xi_shape, ndim, nans, out_of_bounds


    def _validate_grid_dimensions(self, points, method):
        k = self._SPLINE_DEGREE_MAP[method]
        for i, point in enumerate(points):
            ndim = len(np.atleast_1d(point))
            if ndim <= k:
                raise ValueError(f"There are {ndim} points in dimension {i},"
                                 f" but method {method} requires at least "
                                 f" {k+1} points per dimension.")



    def _find_out_of_bounds(self, xi):
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        return out_of_bounds


def interpn(points, values, xi, nWave, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular or rectilinear grids.

    Strictly speaking, not all regular grids are supported - this function
    works on *rectilinear* grids, that is, a rectangular grid with even or
    uneven spacing.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    xi : ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear",

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at `xi`. See notes for behaviour when
        ``xi.ndim == 1``.

    See Also
    --------
    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions
    RegularGridInterpolator : interpolation on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).
    Notes
    -----

    .. versionadded:: 0.14

    In the case that ``xi.ndim == 1`` a new axis is inserted into
    the 0 position of the returned array, values_x, so its shape is
    instead ``(1,) + values.shape[ndim:]``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolation.
    """


    # sanity check 'method' kwarg
    if method not in ["linear", "nearest", "cubic", "quintic", "pchip",
                      "splinef2d", "slinear"]:
        raise ValueError("interpn only understands the methods 'linear', "
                         "'nearest', 'slinear', 'cubic', 'quintic', 'pchip', "
                         f"and 'splinef2d'. You provided {method}.")

    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    ndim = values.ndim


    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))

    # Check the points
    # Grid is the new name of points and (xa, xb)
    grid, descending_dimensions = _check_points(points)

    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         "%d, but this RegularGridInterpolator has "
                         "dimension %d" % (xi.shape[-1], len(grid)))


    if bounds_error:
        for i, p in enumerate(xi.T):
            # Check if the coordinates of xi (ch14) is embeded in the coordinates of grid (xa, xb) (ch4a)
            if not np.logical_and(np.all(grid[i][0] <= p),
                                  np.all(p <= grid[i][-1])):
                raise ValueError("One of the requested xi is out of bounds "
                                 "in dimension %d" % i)

    # perform interpolation
    if method in ["linear"]:
        interp = RegularGridInterpolator(points, values, nWave, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        # Call __call function of class RegularGridInterpolator
        return interp(xi)





# def tmp_test_interpolation_FoV2cube_python_cython():
templates = global_variable_testing.templates
n_lamnda = len(global_variable_testing.wavelength_axis)
im_shape = global_variable_testing.im_shape
local_shape = (im_shape[0]-100, im_shape[1]-100)
cube = np.random.random((n_lamnda, im_shape[0], im_shape[1]))
out_cube = np.zeros_like(cube)

face = misc.face()
face = face[::2,::4,:]
face = face[:im_shape[0], :im_shape[1], 0]
cube[:] = face

# Wavelength index
wavel_idx = np.arange(n_lamnda)

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

# Cube Coordinates
cube_origin_alpha = 0
cube_origin_beta = 0
cube_alpha_axis = np.arange(im_shape[0]).astype(np.float64)* step_Angle.degree
cube_beta_axis = np.arange(im_shape[1]).astype(np.float64)* step_Angle.degree
cube_alpha_axis -= np.mean(cube_alpha_axis)
cube_beta_axis -= np.mean(cube_beta_axis)
cube_alpha_axis += cube_origin_alpha
cube_beta_axis += cube_origin_beta
cube_shape = cube.shape

# Def Channel spec.
ch2a = instru.IFU(
    fov=instru.FOV(2.0/3600, 2.8/3600, origin=instru.Coord(0, 0), angle=45),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=None,
    pce=None,
    wavel_axis=None,
    name="2A",
)

local_alpha_axis, local_beta_axis = ch2a.fov.local_coords(step_Angle.degree,5* step_Angle.degree, 5* step_Angle.degree)
local_alpha_coord, local_beta_coord = (ch2a.fov).local2global(
        local_alpha_axis, local_beta_axis
    )
local_out_shape = (len(wavel_idx),) + local_alpha_coord.shape


local_coords = np.vstack(
        [
            np.repeat(
                np.repeat(wavel_idx.reshape((-1, 1, 1)), local_out_shape[1], axis=1),
                local_out_shape[2],
                axis=2,
            ).ravel(),
            np.repeat(local_alpha_coord[np.newaxis], local_out_shape[0], axis=0).ravel(),
            np.repeat(local_beta_coord[np.newaxis], local_out_shape[0], axis=0).ravel(),
        ]
    ).T


optimized_local_coords = np.vstack(
        [
            local_alpha_coord.ravel(),
            local_beta_coord.ravel()
        ]
    ).T 


gridded_coord = interpn((cube_alpha_axis, 
                        cube_beta_axis), 
                        cube, 
                        optimized_local_coords, 
                        len(wavel_idx))

gridded = cube[:,gridded_coord.astype(int)[:,0], gridded_coord.astype(int)[:,1]].reshape(307, local_out_shape[1], local_out_shape[2])


# Projection
proj_cube = np.zeros_like(cube)
alpha_coord, beta_coord = ch2a.fov.global2local(
                    cube_alpha_axis, cube_beta_axis
                    )
            
optimized_global_coords = np.vstack(
        [
            alpha_coord.ravel(),
            beta_coord.ravel()
        ]
        ).T

gridded_t_coord = interpn( (local_alpha_axis, local_beta_axis), 
                        gridded[0], 
                        optimized_global_coords, 
                        len(wavel_idx),
                        bounds_error=False, 
                        fill_value=0,)

proj_cube = gridded[:,gridded_t_coord[:,0].astype(int), gridded_t_coord[:,1].astype(int)].reshape(307,251,251)


test_cube_alpha_axis = np.tile(cube_alpha_axis, len(cube_beta_axis))
test_cube_beta_axis= np.repeat(cube_beta_axis, len(cube_beta_axis))

# gridata = griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube[0].ravel(), (local_alpha_coord, local_beta_coord))
# degridata = griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
indexes = nearest_neighbor_interpolation.griddata((test_cube_alpha_axis.ravel(), test_cube_beta_axis.ravel()), cube[0].ravel(), (local_alpha_coord, local_beta_coord))
gridata = cube[0].ravel()[indexes].reshape(local_alpha_coord.shape[0], local_alpha_coord.shape[1])

indexes_t = nearest_neighbor_interpolation.griddata((local_alpha_coord.ravel(), local_beta_coord.ravel()), gridata.ravel(), (test_cube_alpha_axis.reshape(251,251), test_cube_beta_axis.reshape(251,251)))
degridata = gridata.ravel()[indexes_t].reshape(251,251)