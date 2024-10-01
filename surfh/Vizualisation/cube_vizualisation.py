import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_cube(cube, wavelength_cube):
    
    if cube.shape[0] != 4:
        idx = np.where(np.sum(cube, axis=(1,2)) != 0)[0]
        nzero_slice = slice(idx[0], idx[-1])
        print(nzero_slice)
        cube = cube[nzero_slice, ...]
        wavelength_cube = wavelength_cube[nzero_slice]

    # Initial lambda index
    initial_lambda = 0
    nLambda = cube.shape[0]

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)  # Adjust the subplot to make space for the slider

    # Display the initial slice
    slice_plot = ax.imshow(cube[initial_lambda, :, :], cmap='viridis')
    ax.set_title(f'Lambda slice: {initial_lambda}')
    
    # Add a colorbar
    cbar = plt.colorbar(slice_plot, ax=ax)

    # Create the slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Lambda', 0, nLambda - 1, valinit=initial_lambda, valstep=1)

    # Update function to be called when the slider is changed
    def update(val):
        lambda_index = int(slider.val)
        # Update the image data
        new_slice = cube[lambda_index, :, :]
        slice_plot.set_data(new_slice)
        
        # Update the color limits for the current slice
        slice_plot.set_clim(vmin=np.min(new_slice), vmax=np.max(new_slice))

        # Redraw the colorbar with the new limits
        cbar.update_normal(slice_plot)

        # Update the title to show the current lambda slice
        ax.set_title(f'Lambda slice: {wavelength_cube[lambda_index]}')
        
        # Redraw the figure canvas
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()