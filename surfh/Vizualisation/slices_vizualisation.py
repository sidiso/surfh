import numpy as np
import matplotlib.pyplot as plt



def visualize_corrected_slices(data_shape, corrected_data):
    """
    Visualize the interpolated image using labeled connected components.
    """
    nslits = data_shape[0]
    space = 0.1
    fig, ax = plt.subplots(figsize=(nslits * 1, 32))
    ax.axis('off')
    vmin, vmax = 0, 25000
    for slit in range(nslits):

        left = slit - 1 + (slit - 1) * space
        right = left + 1
        height = data_shape[1]
        im = ax.imshow(np.flipud(corrected_data[slit]), extent=[left, right, 0, height], aspect='auto',vmin=vmin, cmap='viridis')

    ax.set_xlim(0, nslits + (nslits - 1) * space)
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Intensity')

    plt.show()