import numpy as np
import matplotlib.pyplot as plt



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
