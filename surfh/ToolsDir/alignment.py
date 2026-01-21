import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import shift
from skimage.transform import rotate
import numpy as np


def interactive_align(im1, im2, mask=None):
    """
    Interface interactive avec sliders (rotation, shift X/Y, alpha)
    et un bouton pour interchanger les images (fond / overlay).
    """

    if mask is None:
        mask = np.ones_like(im1)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Mode initial : im2 en fond, im1 superposé
    base_img = ax.imshow(im2, cmap='viridis')
    overlay_img = ax.imshow(im1, cmap='magma', alpha=0.5)

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_angle = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_shiftx = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_shifty = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_alpha  = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

    s_angle = Slider(ax_angle, 'Rotation (°)', -180, 180, valinit=0, valstep=0.5)
    s_shiftx = Slider(ax_shiftx, 'Shift X', -200, 200, valinit=0, valstep=1)
    s_shifty = Slider(ax_shifty, 'Shift Y', -200, 200, valinit=0, valstep=1)
    s_alpha  = Slider(ax_alpha,  'Alpha', 0.0, 1.0, valinit=0.5, valstep=0.05)

    # Bouton toggle transparence
    ax_button = plt.axes([0.025, 0.05, 0.15, 0.05])
    button = Button(ax_button, 'Toggle View', color='lightblue', hovercolor='0.8')

    # État du toggle
    toggle_state = {"swapped": False}

    # Fonction de mise à jour
    def update(val):
        angle = s_angle.val
        dx = s_shiftx.val
        dy = s_shifty.val
        alpha = s_alpha.val

        rotated = rotate(im1, angle, resize=False, order=3, preserve_range=True)
        shifted = shift(rotated, shift=(dy, dx), order=3)

        overlay_img.set_data(shifted)
        overlay_img.set_alpha(alpha)
        fig.canvas.draw_idle()

    # Fonction toggle transparence
    def toggle(event):
        if toggle_state["swapped"]:
            # im2 en fond, im1 en overlay
            base_img.set_data(im2)
            base_img.set_cmap('viridis')
            base_img.set_alpha(1.0)
            overlay_img.set_cmap('magma')
            toggle_state["swapped"] = False
        else:
            # im1 en fond, im2 en overlay
            base_img.set_data(im1)
            base_img.set_cmap('magma')
            base_img.set_alpha(1.0)
            overlay_img.set_data(im2)
            overlay_img.set_cmap('viridis')
            toggle_state["swapped"] = True
        fig.canvas.draw_idle()

    # Lier sliders et bouton
    s_angle.on_changed(update)
    s_shiftx.on_changed(update)
    s_shifty.on_changed(update)
    s_alpha.on_changed(update)
    button.on_clicked(toggle)

    plt.show()
