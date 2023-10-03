__all__ = ['RegularGridInterpolator', 'interpn']

import itertools

import numpy as np
import time

import cythons_files

# OLD Cython

# OLD Cython
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
            index = cythons_files.find_interval_ascending(grid_i,
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



# Old Cython
def evaluate_linear_2d(values, # cannot declare as ::1
                       indices,       # unless prior
                       norm_distances,    # np.ascontiguousarray
                       grid,
                       out):

    
    num_points = indices.shape[1]      # XXX: npy_intp?

    # i0, i1, point
    # y0, y1, result

    # Check if out en num points are equal
    assert out.shape[0] == num_points

    # Check if One of the dimension of the input grid is 1D
    if grid[1].shape[0] == 1:
        # linear interpolation along axis=0
        for point in range(num_points):
            i0 = indices[0, point]
            if i0 >= 0:
                y0 = norm_distances[0, point]
                result = values[i0, 0]*(1 - y0) + values[i0+1, 0]*y0
                out[point] = result
            else:
                # xi was nan: find_interval returns -1
                out[point] = 0
    # Check if One of the dimension of the input grid is 1D
    elif grid[0].shape[0] == 1:
        # linear interpolation along axis=1
        for point in range(num_points):
            i1 = indices[1, point]
            if i1 >= 0:
                y1 = norm_distances[1, point]
                result = values[0, i1]*(1 - y1) + values[0, i1+1]*y1
                out[point] = result
            else:
                # xi was nan: find_interval returns -1
                out[point] = 0
    # None of the dimension of the input grid is 1D
    else:
        # Remind indices shape is (2, X)
        for point in range(num_points):

            # Take the Alpĥa and Beta idexes of the input grid from the output grid
            i0, i1 = indices[0, point], indices[1, point]

            # Verification of non-zero index (Error)
            if i0 >=0 and i1 >=0:

                # Take the norm distance of the closest pixel
                y0, y1 = norm_distances[0, point], norm_distances[1, point]

                result = 0.0
                # Compute the interpolated point following (https://paulbourke.net/miscellaneous/interpolation/   , Trilinear interpolation)
                result = result + values[i0, i1] * (1 - y0) * (1 - y1)
                result = result + values[i0, i1+1] * (1 - y0) * y1
                result = result + values[i0+1, i1] * y0 * (1 - y1)
                result = result + values[i0+1, i1+1] * y0 * y1
                out[point] = result
            else:
                # xi was nan
                out[point] = np.nan

    return np.asarray(out)



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


class RegularGridInterpolator:
    """
    Interpolation on a regular or rectilinear grid in arbitrary dimensions.

    The data must be defined on a rectilinear grid; that is, a rectangular
    grid with even or uneven spacing. Linear, nearest-neighbor, spline
    interpolations are supported. After setting up the interpolator object,
    the interpolation method may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip". This
        parameter will become the default for the object's ``__call__``
        method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True.

    fill_value : float or None, optional
        The value to use for points outside of the interpolation domain.
        If None, values outside the domain are extrapolated.
        Default is ``np.nan``.

    Methods
    -------
    __call__

    Attributes
    ----------
    grid : tuple of ndarrays
        The points defining the regular grid in n dimensions.
        This tuple defines the full grid via
        ``np.meshgrid(*grid, indexing='ij')``
    values : ndarray
        Data values at the grid.
    method : str
        Interpolation method.
    fill_value : float or ``None``
        Use this value for out-of-bounds arguments to `__call__`.
    bounds_error : bool
        If ``True``, out-of-bounds argument raise a ``ValueError``.

    Notes
    -----
    Contrary to `LinearNDInterpolator` and `NearestNDInterpolator`, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    In other words, this class assumes that the data is defined on a
    *rectilinear* grid.

    .. versionadded:: 0.14

    The 'slinear'(k=1), 'cubic'(k=3), and 'quintic'(k=5) methods are
    tensor-product spline interpolators, where `k` is the spline degree,
    If any dimension has fewer points than `k` + 1, an error will be raised.

    .. versionadded:: 1.9

    If the input data is such that dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}
    _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
    _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS

    def __init__(self, points, values, o_shape, nWave, method="linear", bounds_error=True,
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
        self.o_shape = o_shape
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

        indices, norm_distances = cythons_files.find_indices(self.grid, xi.T)

        result = cythons_files.solve_2D_hypercube(indices, norm_distances, self.values, indices.shape[1], self.nWave)
        


        if not self.bounds_error and self.fill_value is not None:
            print("Out_of_bounds shape is ", out_of_bounds.shape)
            print("Res shape" , result.shape)
            result[:,out_of_bounds] = self.fill_value

        # f(nan) = nan, if any
        if np.any(nans):
            result[nans] = np.nan
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


    def _evaluate_linear_optim(self, indices, norm_distances):

        num_points = indices.shape[1]
        
        out = np.empty((self.nWave, indices.shape[1]), dtype=self.values.dtype)
        # The formula for linear interpolation in 2d takes the form:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
                # Remind indices shape is (2, X)
        for point in range(num_points):

            # Take the Alpĥa and Beta idexes of the input grid from the output grid
            i0, i1 = indices[0, point], indices[1, point]

            # Verification of non-zero index (Error)
            if i0 >=0 and i1 >=0:

                # Take the norm distance of the closest pixel
                y0, y1 = norm_distances[0, point], norm_distances[1, point]

                result = 0.0
                # Compute the interpolated point following (https://paulbourke.net/miscellaneous/interpolation/   , Trilinear interpolation)
                result = result + self.values[:,i0, i1] * (1 - y0) * (1 - y1)
                if point == 0:
                    print(y0)
                result = result + self.values[:,i0, i1+1] * (1 - y0) * y1
                if point == 0:
                    print(y1)
                result = result + self.values[:,i0+1, i1] * y0 * (1 - y1)
                if point == 0:
                    print((1 - y0))
                result = result + self.values[:,i0+1, i1+1] * y0 * y1
                if point == 0:
                    print((1-y1))
                out[:,point] = result
            else:
                print("ERRORRRRRRRR : NAN")
                # xi was nan
                #out[point] = np.nan      
       
        return  out


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
        print("SHAPE GRID is ", len(self.grid), len(self.grid[0]), xi.shape)
        for x, grid in zip(xi, self.grid):
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        print(out_of_bounds, out_of_bounds.shape)
        return out_of_bounds


def interpn(points, values, xi, o_shape, nWave, method="linear", bounds_error=True,
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
        interp = RegularGridInterpolator(points, values, o_shape, nWave, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        # Call __call function of class RegularGridInterpolator
        return interp(xi)
