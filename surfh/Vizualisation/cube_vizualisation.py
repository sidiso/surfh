import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_cube(cube, wavelength_cube):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if cube.shape[0] != 4:
        idx = np.where(np.sum(cube, axis=(1, 2)) != 0)[0]
        nzero_slice = slice(idx[0], idx[-1])
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
    ax.set_title(f'Lambda slice: {wavelength_cube[initial_lambda]}')

    # Add a colorbar
    cbar = plt.colorbar(slice_plot, ax=ax)

    # Create the slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Lambda', 0, nLambda - 1, valinit=initial_lambda, valstep=1)

    # Update function to be called when the slider is changed
    def update(val):
        lambda_index = int(slider.val)
        new_slice = cube[lambda_index, :, :]

        # Update the image data
        slice_plot.set_data(new_slice)

        # Update the color limits dynamically
        vmin, vmax = np.nanmin(new_slice), np.nanmax(new_slice)
        slice_plot.set_clim(vmin, vmax)

        # Redraw the colorbar (uses the ScalarMappable limits)
        cbar.mappable.set_clim(vmin, vmax)
        cbar.draw_all()

        # Update the title
        ax.set_title(f'Lambda slice: {wavelength_cube[lambda_index]}')

        # Redraw the canvas
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()





def plot_two_cubes(cube_fusion, wavelength_fusion, cube_pipeline, wavelength_pipeline):
    
    """
    Cube1 is the fusion cube.
    Cube2 is the pipeline cube.
    We took nearest slice of pipeline cube to match fusion cube.
    """

    # viz_pipeline_cube = np.empty_like(cube_fusion)

    # Fonction pour réduire le vecteur A
    def reduce_vector(A, coords_A, coords_B):
        reduced_A = []
        for coord in coords_B:
            # Trouver l'indice de la coordonnée la plus proche
            closest_idx = np.argmin(np.abs(coords_A - coord))
            # Ajouter la valeur correspondante
            reduced_A.append(A[closest_idx])
        return np.array(reduced_A)

    viz_pipeline_cube = reduce_vector(cube_pipeline, wavelength_pipeline, wavelength_fusion)


    # Initial lambda index
    initial_lambda = wavelength_fusion[0]
    nLambda = cube_fusion.shape[0]

    # Create a figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.subplots_adjust(left=0.1, bottom=0.25)  # Adjust the subplot to make space for the slider

    # Display the initial slice
    slice_plot1 = axes[0].imshow(cube_fusion[0, :, :], cmap='viridis')
    slice_plot2 = axes[1].imshow(viz_pipeline_cube[0, :, :], cmap='viridis')

    axes[0].set_title(f'Lambda = {str(round(initial_lambda, 2))}', fontsize=30)
    axes[1].set_title(f'Lambda = {str(round(initial_lambda, 2))}', fontsize=30)

    
    # Add a colorbar
    cbar = plt.colorbar(slice_plot1, ax=axes[1])
    cbar = plt.colorbar(slice_plot1, ax=axes[0])

    # Add a colorbar
    # cbar = plt.colorbar(slice_plot2, ax=axes[1])

    # Create the slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Lambda', 0, nLambda - 1, valinit=0, valstep=1)

    # Update function to be called when the slider is changed
    def update(val):
        lambda_index = int(slider.val)
        # Update the image data
        new_slice1 = cube_fusion[lambda_index, :, :]
        slice_plot1.set_data(new_slice1)

        new_slice2 = viz_pipeline_cube[lambda_index, :, :]
        slice_plot2.set_data(new_slice2)
        
        # Update the color limits for the current slice
        slice_plot1.set_clim(vmin=np.nanmin(new_slice1), vmax=np.nanmax(new_slice1))
        slice_plot2.set_clim(vmin=np.nanmin(new_slice2), vmax=np.nanmax(new_slice2))

        # Redraw the colorbar with the new limits
        # cbar.update_normal(slice_plot1)

        # Update the title to show the current lambda slice
        axes[0].set_title(f'Lambda = {str(round(wavelength_fusion[lambda_index], 2))}', fontsize=30)
        axes[1].set_title(f'Lambda = {str(round(wavelength_fusion[lambda_index], 2))}', fontsize=30)
        
        # Redraw the figure canvas
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()


def plot_concatenated_cubes(cubes_list, wavelength_cubes_list):
    # Step 1: Concatenate all cubes along the spectral dimension (axis 0)
    cube = np.concatenate(cubes_list, axis=0)
    wavelength_cube = np.concatenate(wavelength_cubes_list)
    
    # Step 2: Handle case if there are zero slices
    if cube.shape[0] != 4:
        idx = np.where(np.sum(cube, axis=(1, 2)) != 0)[0]
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


def plot_maps(estimated_maps):
    nrow = 2#estimated_maps.shape[0] // 2
    ncols = estimated_maps.shape[0] // 2
    print(nrow)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncols, sharex = True, sharey = True)

    for i in range(nrow):
        for j in range(ncols):
            print(i,j)
            m = axes[i,j].imshow(estimated_maps[i*ncols+j])
            fig.colorbar(m, ax=axes[i,j])
