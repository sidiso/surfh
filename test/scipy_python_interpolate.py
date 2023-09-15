__all__ = ['RegularGridInterpolator', 'interpn']

import itertools

import numpy as np

# OLD Cython
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
    interval = prev_interval

    #First value of x
    a = x[0]

    #Last value of x
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

# OLD Cython
def find_indices(grid, xi):

    # Axes to iterate over
    I = xi.shape[0] # = 2 because we took the Transpose of xi
    J = xi.shape[1] # = 66203 because we took the Transpose of xi
    print("There is J = ", J)
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


# OLD Cython
def _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.

    """
    # j, n

    print("Before check, points is ", type(points), len(points), points.shape)
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
    print("After check, points is ", type(points), len(points), points.shape)
    return points



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


def _check_dimensionality(points, values):
    if len(points) > values.ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), values.ndim))
    for i, p in enumerate(points):
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))

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

    def __init__(self, points, values, method="linear", bounds_error=True,
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

        #Check if "im" and xa,xb have same coordinate dimension
        self._check_dimensionality(self.grid, self.values)

        # Set fill value to np.nan
        self.fill_value = self._check_fill_value(self.values, fill_value)
        if self._descending_dimensions:
            self.values = np.flip(values, axis=self._descending_dimensions)

    def _check_dimensionality(self, grid, values):
        _check_dimensionality(grid, values)

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

        if method == "linear":
            # Return | indices : Array of the same time of Xi, where each element is the idx of the input 
            #                    grid corresponding to the element with the closest coordinates of the output grid.
            #        | norm_distances : Same as indices but the element in the array is the distance
            indices, norm_distances = find_indices(self.grid, xi.T)

            if (ndim == 2 and hasattr(self.values, 'dtype') and
                    self.values.ndim == 2 and self.values.flags.writeable and
                    self.values.dtype in (np.float64, np.complex128) and
                    self.values.dtype.byteorder == '='):
                # until cython supports const fused types, the fast path
                # cannot support non-writeable values
                # a fast path

                # out is a vector of the size of the number of element is the output grid
                out = np.empty(indices.shape[1], dtype=self.values.dtype)

                print("Going to do evaluate_linear_2d")
                result = evaluate_linear_2d(self.values,
                                            indices,
                                            norm_distances,
                                            self.grid,
                                            out)
            else:
                print("Going to do _evaluate_linear")
                result = self._evaluate_linear(indices, norm_distances)


        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        # f(nan) = nan, if any
        if np.any(nans):
            result[nans] = np.nan
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _prepare_xi(self, xi):
        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
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

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))
        print("Vslice is ", vslice)

        np.save('norm_distances.npy', norm_distances)

        # Compute shifting up front before zipping everything together
        shift_norm_distances = [1 - yi for yi in norm_distances]
        shift_indices = [i + 1 for i in indices]

        # The formula for linear interpolation in 2d takes the form:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
        zipped1 = zip(indices, shift_norm_distances)
        zipped2 = zip(shift_indices, norm_distances)
        
        # Take all products of zipped1 and zipped2 and iterate over them
        # to get the terms in the above formula. This corresponds to iterating
        # over the vertices of a hypercube.
        hypercube = itertools.product(*zip(zipped1, zipped2))
        value = np.array([0.])
        for h in hypercube:
            edge_indices, weights = zip(*h)
            weight = np.array([1.])
            for w in weights:
                weight = weight * w
            term = np.asarray(self.values[edge_indices]) * weight[vslice]
            value = value + term   # cannot use += because broadcasting
        return value

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


def interpn(points, values, xi, method="linear", bounds_error=True,
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
    _check_dimensionality(grid, values)

    # sanity check requested xi
    # return xi as a np.array
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
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
        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        # Call __call function of class RegularGridInterpolator
        return interp(xi)
