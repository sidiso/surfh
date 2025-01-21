import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



def visualize_corrected_slices(data_shape, corrected_data):
    """
    Visualize the interpolated image using labeled connected components.
    """
    nslits = data_shape[0]
    print(f"There are {nslits} slits")
    space = 0.1  # Space between the slices

    # Create a figure, adjusting size based on the number of slits
    fig, ax = plt.subplots(figsize=((nslits + (nslits - 1) * space), 32))
    ax.axis('off')  # Hide the axes for a clean image

    vmin, vmax = 0, 25000  # Define min/max values for color scaling

    for slit in range(nslits):
        print(f'Slice {slit}')
        
        # Positioning each slice with the correct spacing
        left = slit + slit * space
        right = left + 1
        height = data_shape[1]

        # Plot the corrected slice
        im = ax.imshow(np.flipud(corrected_data[slit]), extent=[left, right, 0, height], 
                       aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    # Set the x-axis limits to fit all slices
    ax.set_xlim(0, nslits + (nslits - 1) * space)

    # Add a colorbar with an intensity label
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Intensity')

    plt.show()


def visualize_projected_corrected_slices(slices, wavel_idx):

    plt.imshow(slices[:,wavel_idx,:])
    plt.colorbar()
    plt.show()


def visualize_projected_slices(slices, wavels=None):

    slices = slices.transpose(1,0,2)
    # Initial lambda index
    initial_lambda = 0
    nLambda = slices.shape[0]

    if wavels is None:
        wavels = np.arange(nLambda)

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)  # Adjust the subplot to make space for the slider

    # Display the initial slice
    slice_plot = ax.imshow(slices[initial_lambda, :, :], cmap='viridis')
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
        new_slice = slices[lambda_index, :, :]
        slice_plot.set_data(new_slice)
        
        # Update the color limits for the current slice
        slice_plot.set_clim(vmin=np.min(new_slice), vmax=np.max(new_slice))

        # Redraw the colorbar with the new limits
        cbar.update_normal(slice_plot)

        # Update the title to show the current lambda slice
        ax.set_title(f'Lambda slice: {wavels[lambda_index]}')
        
        # Redraw the figure canvas
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()
