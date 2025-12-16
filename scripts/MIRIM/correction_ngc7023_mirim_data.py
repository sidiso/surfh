import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

mirim_data_dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/Raw_MIRIM/'
filter_img = mirim_data_dir + 'Level3_F1000W_i2d_aligned.fits'

# --- Load FITS ---
with fits.open(filter_img) as hdul:
    data = hdul[1].data.copy()
    header = hdul[1].header

print(f"Shape data = {data.shape}")
data = data[470:-15, 370:-44]
print(f"Shape data = {data.shape}")

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='viridis')
plt.colorbar(im, ax=ax)

# -------- Patch function ----------
def apply_patch(xmin, xmax, ymin, ymax):
    global data

    xmin, xmax = int(xmin), int(xmax)
    ymin, ymax = int(ymin), int(ymax)

    border_pixels = []
    if xmin > 0:
        border_pixels.append(data[ymin:ymax, xmin-1])
    if xmax < data.shape[1]-1:
        border_pixels.append(data[ymin:ymax, xmax+1])
    if ymin > 0:
        border_pixels.append(data[ymin-1, xmin:xmax])
    if ymax < data.shape[0]-1:
        border_pixels.append(data[ymax+1, xmin:xmax])

    border_pixels = np.concatenate(border_pixels)
    fill_value = np.median(border_pixels)

    data[ymin:ymax, xmin:xmax] = fill_value
    im.set_data(data)
    fig.canvas.draw_idle()

# -------- Rectangle selector callback --------
def onselect(eclick, erelease):
    xmin, xmax = sorted([eclick.xdata, erelease.xdata])
    ymin, ymax = sorted([eclick.ydata, erelease.ydata])
    apply_patch(xmin, xmax, ymin, ymax)

# --- RectangleSelector compatible ---
toggle_selector = RectangleSelector(
    ax,
    onselect,
    interactive=True
)

plt.show()
