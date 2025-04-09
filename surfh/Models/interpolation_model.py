

from aljabr.linop import Shape
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix

from aljabr import LinOp, dottest

class InterpolationLinOp(LinOp):
    def __init__(self,  ishape, 
                        oshape, 
                        data_coords,
                        interp_coords,
                        ):
        super().__init__(ishape, oshape, '_', np.float64)
        self.weights, self.S_op = self._compute_weights(data_coords, interp_coords)
        self.data_coords = data_coords
        self.interp_coords = interp_coords


    def _compute_weights(self, data_coords, interp_coords):
        weights = []
        S_op = np.zeros((len(interp_coords), len(data_coords)))
        for idx, x in enumerate(interp_coords):
            # Find the interval [x_old[i], x_old[i+1]] that contains x
            i = np.searchsorted(data_coords, x) - 1
            i = np.clip(i, 0, len(data_coords) - 2)
            x1, x2 = data_coords[i], data_coords[i + 1]
            # Linear interpolation weights
            w1 = (x2 - x) / (x2 - x1)
            w2 = (x - x1) / (x2 - x1)
            weights.append((i, w1, w2))
            S_op[idx, i] = w1
            S_op[idx, i+1] = w2
        return weights, S_op

    def forward(self, global_cube: np.ndarray) -> np.ndarray:
        """Interpolates from x_old to x_new."""
        result = np.zeros(len(self.interp_coords))
        for idx, (i, w1, w2) in enumerate(self.weights):
            result[idx] = w1 * global_cube[i] + w2 * global_cube[i + 1]
        return result
    
    def adjoint(self, local_cube: np.ndarray) -> np.ndarray:
        """Computes the adjoint operation."""
        result = np.zeros(len(self.data_coords))
        for idx, (i, w1, w2) in enumerate(self.weights):
            result[i] += w1 * local_cube[idx]
            result[i + 1] += w2 * local_cube[idx]
        return result



class InterpolationOperator(LinearOperator):
    def __init__(self, x_old, x_new):
        self.x_old = np.asarray(x_old)
        self.x_new = np.asarray(x_new)
        self.shape = (len(x_new), len(x_old))
        self.dtype = np.float64

        # Compute interpolation weights
        self.weights = self._compute_weights()

    def _compute_weights(self):
        weights = []
        for x in self.x_new:
            # Find the interval [x_old[i], x_old[i+1]] that contains x
            i = np.searchsorted(self.x_old, x) - 1
            i = np.clip(i, 0, len(self.x_old) - 2)
            x1, x2 = self.x_old[i], self.x_old[i + 1]
            # Linear interpolation weights
            w1 = (x2 - x) / (x2 - x1)
            w2 = (x - x1) / (x2 - x1)
            weights.append((i, w1, w2))
        return weights

    def _matvec(self, v):
        """Interpolates from x_old to x_new."""
        result = np.zeros(len(self.x_new))
        for idx, (i, w1, w2) in enumerate(self.weights):
            result[idx] = w1 * v[i] + w2 * v[i + 1]
        return result

    def _rmatvec(self, w):
        """Computes the adjoint operation."""
        result = np.zeros(len(self.x_old))
        for idx, (i, w1, w2) in enumerate(self.weights):
            result[i] += w1 * w[idx]
            result[i + 1] += w2 * w[idx]
        return result



class Interpolation2DLinOp(LinOp):
    def __init__(self,  ishape, 
                        oshape, 
                        data_coords,
                        interp_coords,
                        ):
        super().__init__(ishape, oshape, '_', np.float64)
        self.S_op = self._compute_weights(data_coords, interp_coords)
        self.S_op_sparse = self._compute_weights_sparse(data_coords, interp_coords)
        self.data_coords = data_coords # Tupple of shape (2, Px) avec Px = Nx * Nx
        self.interp_coords = interp_coords # Tupple of shape (2, Py) avec Py = Ny * Ny

    def _compute_weights_sparse(self, data_coords, interp_coords):
        sparse_weight = np.zeros((3, ))
        return

    def _compute_weights(self, data_coords, interp_coords):
        S_op = np.zeros((len(interp_coords[0]), len(data_coords[0])))

        y_coords, x_coords = data_coords
        N = self.ishape[0]

        for idx, n_coords in enumerate(interp_coords.T):
            x = n_coords[1]
            y = n_coords[0]

            i = np.searchsorted(x_coords, x) - 1
            i = np.clip(i, 0, len(x_coords) - 2)

            # Find the interval [y_old[j], y_old[j+1]] that contains y
            j = np.searchsorted(y_coords[i:i+N], y) - 1
            j = np.clip(j, 0, N - 2)

            if (i+j+1)%N == 0:
                j = j - 1
            if (i+N+1) > len(x_coords)-1:
                i = i - N
            # print(f"i: {i}, j: {j}")
            # print(f"x: {x}, y: {y}")

            x1, x2 = x_coords[i+j], x_coords[i+j+N]
            y1, y2 = y_coords[i+j], y_coords[i+j+1]

            denom = ((x2 - x1) * (y2 - y1))
            # print(f"x1: {x1}, x2: {x2}")
            # print(f"y1: {y1}, y2: {y2}")
            # print(f"denom: {denom}")

            # Bilinear interpolation weights
            w1 = ((x2 - x) * (y2 - y)) / ((x2 - x1) * (y2 - y1))
            w2 = ((x - x1) * (y2 - y)) / ((x2 - x1) * (y2 - y1))
            w3 = ((x2 - x) * (y - y1)) / ((x2 - x1) * (y2 - y1))
            w4 = ((x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))

            # print(f"Sum weights: {w1 + w2 + w3 + w4}")
            # Flatten the indices for the weight matrix
            flat_idx1 = i + j
            flat_idx2 = i + N + j
            flat_idx3 = i + j + 1
            flat_idx4 = i + N + j + 1

            # print(f"flat_idx1: {flat_idx1}, flat_idx2: {flat_idx2}, flat_idx3: {flat_idx3}, flat_idx4: {flat_idx4}")

            S_op[idx, flat_idx1] = w1
            S_op[idx, flat_idx2] = w2
            S_op[idx, flat_idx3] = w3
            S_op[idx, flat_idx4] = w4
        return S_op
    

    def _compute_weights_sparse(self, data_coords, interp_coords):
        rows = []
        cols = []
        data = []

        y_coords, x_coords = data_coords
        N = self.ishape[0]

        for idx, n_coords in enumerate(interp_coords.T):
            x = n_coords[1]
            y = n_coords[0]

            i = np.searchsorted(x_coords, x) - 1
            i = np.clip(i, 0, len(x_coords) - 2)

            # Find the interval [y_old[j], y_old[j+1]] that contains y
            j = np.searchsorted(y_coords[i:i+N], y) - 1
            j = np.clip(j, 0, N - 2)

            if (i+j+1) % N == 0:
                j = j - 1
            if (i+N+1) > len(x_coords) - 1:
                i = i - N

            x1, x2 = x_coords[i+j], x_coords[i+j+N]
            y1, y2 = y_coords[i+j], y_coords[i+j+1]

            denom = ((x2 - x1) * (y2 - y1))

            # Bilinear interpolation weights
            w1 = ((x2 - x) * (y2 - y)) / denom
            w2 = ((x - x1) * (y2 - y)) / denom
            w3 = ((x2 - x) * (y - y1)) / denom
            w4 = ((x - x1) * (y - y1)) / denom

            # Flatten the indices for the weight matrix
            flat_idx1 = i + j
            flat_idx2 = i + N + j
            flat_idx3 = i + j + 1
            flat_idx4 = i + N + j + 1

            # Ensure the indices are within bounds
            if flat_idx1 >= len(data_coords[0]) or flat_idx2 >= len(data_coords[0]) or flat_idx3 >= len(data_coords[0]) or flat_idx4 >= len(data_coords[0]):
                raise IndexError(f"Index out of bounds: flat_idx1={flat_idx1}, flat_idx2={flat_idx2}, flat_idx3={flat_idx3}, flat_idx4={flat_idx4}")

            # Append the indices and weights to the lists
            rows.extend([idx, idx, idx, idx])
            cols.extend([flat_idx1, flat_idx2, flat_idx3, flat_idx4])
            data.extend([w1, w2, w3, w4])

        # Create the sparse matrix
        S_op_sparse = csr_matrix((data, (rows, cols)), shape=(len(interp_coords[0]), len(data_coords[0])))
        return S_op_sparse


    def forward(self, global_cube: np.ndarray) -> np.ndarray:
        """Interpolates from x_old to x_new."""
        print(f"Shape bots operators self.S_op: {self.S_op.shape}, global_cube: {global_cube.ravel().shape}")
        result = np.matmul(self.S_op, global_cube.ravel())
        return result.reshape(self.oshape)
    
    def forward_sparse(self, global_cube):
        result = np.zeros(len(self.interp_coords[0]))

        # Calcul explicite avec des boucles for
        for i in range(len(self.interp_coords[0])):
            for j in range(self.S_op_sparse.indptr[i], self.S_op_sparse.indptr[i+1]):
                col = self.S_op_sparse.indices[j]
                value = self.S_op_sparse.data[j]
                result[i] += value * global_cube[col]
        return result.reshape(self.oshape)

    def adjoint(self, local_cube: np.ndarray) -> np.ndarray:
        """Computes the adjoint operation."""
        result = np.matmul(self.S_op.T, local_cube.ravel())
        return result.reshape(self.ishape)


# Example usage
x_old = np.linspace(0, 10, 5)
x_new = np.linspace(0, 10, 10)
interp_op = InterpolationOperator(x_old, x_new)
interp_linop = InterpolationLinOp(x_old.shape, x_new.shape, x_old, x_new)

v = np.sin(x_old)
interpolated_v = interp_op @ v
recovered_v = interp_op._rmatvec(interpolated_v)

# Verify the adjoint property numerically
w = np.random.rand(len(x_new))
lhs = np.dot(interp_op @ v, w)      # < A v, w >
rhs = np.dot(v, interp_op._rmatvec(w))  # < v, A* w >

print(f"interp_op LHS ( < A v, w > ) : {lhs}")
print(f"interp_op RHS ( < v, A* w > ): {rhs}")
print(f"interp_op Difference: {abs(lhs - rhs)}")

dottest(interp_linop, echo=True)
res1 = interp_linop.forward(v)
res2 = np.matmul(interp_linop.S_op, v)
print(f"interp_linop Difference: {np.mean(abs(res1 - res2))}")

resT1 = interp_linop.adjoint(res1)
resT2 = np.matmul(interp_linop.S_op.T, res1)
print(f"interp_linop Adjoint Difference: {np.mean(abs(resT1 - resT2))}")



# 2D Interpolation
print("2D Interpolation")
N = 5
P = N*N
Nn = 7
Pn = Nn*Nn
x = np.arange((N*N)).reshape(N,N)

x_coord = np.linspace(2, 7, N)
y_coord = np.linspace(2, 7, N)
coords = np.meshgrid(x_coord, y_coord)
coords = np.stack((coords[0].ravel(), coords[1].ravel()), axis=0)
print(coords.shape)

nx_coord = np.linspace(0, 10, Nn)
ny_coord = np.linspace(0, 10, Nn)
n_coords = np.meshgrid(nx_coord, ny_coord)
n_coords = np.stack((n_coords[0].ravel(), n_coords[1].ravel()), axis=0)

interpn_2D = Interpolation2DLinOp(ishape=x.shape,
                                  oshape=(Nn, Nn),
                                  data_coords=coords,
                                  interp_coords=n_coords)

dottest(interpn_2D, echo=True)



# Example usage
import scipy.misc
import matplotlib.pyplot as plt
from surfh.Models import instru, wavelength_mrs

im = scipy.misc.ascent()
im = im[::4, ::4]

print("###############################")
print("###############################")

print("Create MRS model")
res1a = np.mean([3320, 3710])
spec_blur_1a = instru.SpectralBlur(res1a)
ch1a = instru.IFU(
    instru.FOV(3.2, 3.7, origin=instru.Coord(1.5, -1.5), angle=8.4),
    0.196,
    21,
    spec_blur_1a,
    None,
    wavelength_mrs.get_mrs_wavelength('1a'),
    "1A",
)
res4a = np.mean([1460, 1930])
spec_blur_4a = instru.SpectralBlur(res4a)
ch4a = instru.IFU(
    instru.FOV(6.9, 7.9, origin=instru.Coord(0, 0), angle=0),
    0.273,
    12,
    spec_blur_4a,
    None,
    wavelength_mrs.get_mrs_wavelength('4a'),
    "4A",
)

# Array of alpha coordinate regarding the ch4a FoV and the number of pixel in the image "im" (Here 512)
xa = (
    np.linspace(
        -ch4a.fov.alpha_width / 2, ch4a.fov.alpha_width / 2, im.shape[0]
    )
    + ch4a.fov.origin.alpha
)

# Array of Beta coordinate regarding the ch4a Fov and the number of pixel in the image "im" (Here 512)
xb = (
    np.linspace(
        -ch4a.fov.beta_width / 2, ch4a.fov.beta_width / 2, im.shape[1]
    )
    + ch4a.fov.origin.beta
)
print(f"xa and xb created, shape are {xa.shape} and {xb.shape}")

# Alpha distance between two pixels
step = np.mean(np.diff(xa))
fov = ch1a.fov
# The FoV is rotated by 8°
fov.angle = 8
# As ch1a is smaller than ch4a, na and nb is a 2D array smaller than (512,512), Here (239,277) 
na, nb = fov.coords(step)
print(f"na and nb created, shape are {na.shape} and {nb.shape}")
coords = np.meshgrid(xa, xb)
coords = np.stack((coords[0].ravel(), coords[1].ravel()), axis=0)

n_coords = np.stack((na.ravel(), nb.ravel()), axis=0)


MRS_interpn_2D = Interpolation2DLinOp(ishape=im.shape,
                                    oshape=na.shape,
                                    data_coords=coords,
                                    interp_coords=n_coords)

print(f"Dottest interpn {dottest(MRS_interpn_2D, echo=True)}")

fw_im = MRS_interpn_2D.forward(im)
# fw_im_sparse = MRS_interpn_2D.forward_sparse(im)
# fw_im_sparse = MRS_interpn_2D.S_op_sparse.dot(im)
# print(f"Is the sparse and non sparse equal ? {np.allclose(fw_im, fw_im_sparse)}")

ad_im = MRS_interpn_2D.adjoint(fw_im)

# Plotting
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im1 = axs[0].imshow(im, extent=(xb[0], xb[-1], xa[-1], xa[0]), cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[0].plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    color="white",
)
axs[0].plot(ch1a.fov.origin.alpha, ch1a.fov.origin.alpha, "x")
fig.colorbar(im1, ax=axs[0])


im2 = axs[1].imshow(fw_im, cmap='gray')
axs[1].set_title('Interpolated Image')
axs[1].axis('off')
fig.colorbar(im2, ax=axs[1])


plt.figure()
plt.imshow(ad_im)
plt.colorbar()


plt.show()


"""
##########################################
##########################################
##########################################
"""
