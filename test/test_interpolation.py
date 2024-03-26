import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib import transforms

from surfh.ToolsDir import cython_2D_interpolation 
from surfh.Models import smallmiri as mrs

im = scipy.misc.ascent()


# Array of alpha coordinate regarding the ch4a FoV and the number of pixel in the image "im" (Here 512)
xa = (
    np.linspace(
        -mrs.ch4a.fov.alpha_width / 2, mrs.ch4a.fov.alpha_width / 2, im.shape[0]
    )
    + mrs.ch4a.fov.origin.alpha
)


# Array of Beta coordinate regarding the ch4a Fov and the number of pixel in the image "im" (Here 512)
xb = (
    np.linspace(
        -mrs.ch4a.fov.beta_width / 2, mrs.ch4a.fov.beta_width / 2, im.shape[1]
    )
    + mrs.ch4a.fov.origin.beta
)

# Alpha distance between two pixels
step = np.mean(np.diff(xa))

# We define FoV as ch1a
fov = mrs.ch1a.fov
# The FoV is rotated by 8°
fov.angle = 8

# Alpha and beta coordinate of ch1a with distance "step" between each pixel
# As ch1a is smaller than ch4a, na and nb is a 2D array smaller than (512,512), Here (239,277) 
na, nb = fov.coords(step)

# xa.reshape((-1, 1)) : reshape xa to be a 2D array (512, 1), instead of a vector
# (1, len(xb)) : is a tuple =(1,512)
# np.tile(A, reps) : Construct an array by repeating A the number of times given by reps. Here, repeat "xa" len(xb) times. The result is a (512,512) array
# X.ravel() : Vectorize an Array
# xo : A 2D array where each 2 elements row is the Alpha and Beta coordinates of the pixel in ch4a FoV.
xo = np.vstack(
    [
        np.tile(xa.reshape((-1, 1)), (1, len(xb))).ravel(),
        np.tile(xb.reshape((1, -1)), (len(xa), 1)).ravel(),
    ]
).T

# xi : A 2D array where each 2 elements row is the Alpha and Beta coordinates of the pixel in ch1a FoV.
xi = np.vstack(
    [
        np.asarray(na).ravel(),
        np.asarray(nb).ravel(),
    ]
).T

# Interpolate 
nim = cython_2D_interpolation.interpn((xa, xb), im, xi).reshape(na.shape)

plt.figure(1)
plt.clf()
plt.subplot(2, 2, 1)
plt.title("ch4a FoV Image with black rectangle that illustrate ch1a FoV")
plt.imshow(im, extent=(xb[0], xb[-1], xa[-1], xa[0]))
plt.plot(xi[:, 1], xi[:, 0], ".", color="black", alpha=0.1)
plt.plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    color="white",
)
plt.subplot(2, 2, 2)
plt.title("Interpolated ch1a image, in ch1a reference frame")
plt.imshow(nim)
plt.subplot(2, 2, 3)
plt.title("Interpolated ch1a image, in ch4a reference frame")
plt.imshow(
    nim,
    transform=transforms.Affine2D().rotate_deg(-fov.angle) + plt.gca().transData,
)
plt.subplot(2, 2, 4)
plt.plot(xo[:, 0], xo[:, 1], ".", alpha=0.2)
plt.plot(xi[:, 0], xi[:, 1], ".", alpha=0.2)


plt.show()




"""
def value_func_2D(x,y):
    return 2*x + 3*y +2

def custom_interpolation(grid_in, values_in, grid_out):
    #values_out = np.zeros(grid_out.shape[0])    
    value_out = values_in[0,0]*(grid_in[0][1]-grid_out[0])*(grid_in[1][1]-grid_out[1]) +\
                values_in[1,0]*grid_out[0]*(grid_in[1][1]-grid_out[1]) +\
                values_in[0,1]*(grid_in[0][1]-grid_out[0])*grid_out[1] +\
                values_in[1,1]*grid_out[0]*grid_out[1] 
    return value_out

x = np.linspace(0, 1, 2)
y = np.linspace(0, 1, 2)

grid_in = (x,y)
values_in = value_func_2D(*np.meshgrid(*grid_in))

grid_out = np.random.rand(100,2)

value_out_1 = sp.interpolate.interpn(grid_in, values_in, grid_out)
value_out_2 = custom_interpolation(grid_in, values_in, grid_out.T)

print("Scipu and Custom interpolation are equal ? ", np.allclose(value_out_1, value_out_2))
"""