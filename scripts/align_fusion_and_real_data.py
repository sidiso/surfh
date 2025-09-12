import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from skimage.transform import rescale
from scipy.ndimage import shift
from skimage.transform import rotate
import scipy
from matplotlib.widgets import Slider, Button



def interactive_align(mirim_rescaled, adj, mask):
    """
    Interface interactive avec sliders (rotation, shift X/Y, alpha)
    et un bouton pour interchanger les images (fond / overlay).
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Mode initial : adj en fond, mirim_rescaled superposé
    base_img = ax.imshow(adj, cmap='viridis')
    overlay_img = ax.imshow(mirim_rescaled*mask, cmap='magma', alpha=0.5)

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

        rotated = rotate(mirim_rescaled, angle, resize=False, order=3, preserve_range=True)
        shifted = shift(rotated, shift=(dy, dx), order=3)

        overlay_img.set_data(shifted*mask)
        overlay_img.set_alpha(alpha)
        fig.canvas.draw_idle()

    # Fonction toggle transparence
    def toggle(event):
        if toggle_state["swapped"]:
            # adj en fond, mirim_rescaled en overlay
            base_img.set_data(adj)
            base_img.set_cmap('viridis')
            base_img.set_alpha(1.0)
            overlay_img.set_cmap('magma')
            toggle_state["swapped"] = False
        else:
            # mirim_rescaled en fond, adj en overlay
            base_img.set_data(mirim_rescaled*mask)
            base_img.set_cmap('magma')
            base_img.set_alpha(1.0)
            overlay_img.set_data(adj)
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



dir = '/home/nmonnier/Projects/JWST/hyper-vista/simulation'
mrs_hdul = fits.open(dir + '/NGC7023_ChannelCube_ch1-2-3-4-shortmediumlong_s3d.fits')
mrs_header = mrs_hdul[1].header 
mrs_data = mrs_hdul[1].data

print(f'MRS data shape = {mrs_data.shape}')

res_pix = np.abs(mrs_header['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
print(mrs_header['CDELT1'])
print(mrs_header['CDELT1']*3600)
print(f"Pixel scale = {res_pix} arcsec/pixel")
scale_factor = (res_pix / 0.1, res_pix / 0.1)

mrs_slice = mrs_data[6517]
mrs_slice[np.isnan(mrs_slice)] = 0

# Nombre de pixels à ajouter
pad_y = 125 - mrs_slice.shape[0]  # = 10
pad_x = 125 - mrs_slice.shape[1]  # = 10

pad_top = pad_y // 2
pad_bottom = pad_y - pad_top
pad_left = pad_x // 2
pad_right = pad_x - pad_left

mrs_padded = np.pad(
    mrs_slice,
    pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
    mode='constant',
    constant_values=0
)

fusion_hdul = fits.open(dir + '/res_CubeMask.fits')
fusion_header = fusion_hdul[0].header
fusion_data = fusion_hdul[0].data
fusion_mask = fusion_hdul['MASKS'].data

fusion_data_masked = fusion_data[6517] * fusion_mask[6]

norm_fusion = fusion_data_masked/np.max(fusion_data_masked)
norm_mrs = mrs_padded/np.max(mrs_padded)


def chi2(fusion_img, mrs_img, angle, y_shift, x_shift):
    fusion_img_rot = rotate(fusion_img, angle=angle, resize=False, order=3, preserve_range=True)  
    fusion_img_shift = shift(fusion_img_rot, shift=(y_shift, x_shift), order=3)  
    fusion_img_masked = fusion_img_shift
    fusion_img_masked = fusion_img_masked
    return np.sum((fusion_img_masked - mrs_img)**2)

def get_result_image(fusion_img, angle, y_shift, x_shift):
    fusion_img_rot = rotate(fusion_img, angle=angle, resize=False, order=3, preserve_range=True)
    fusion_img_shift = shift(fusion_img_rot, shift=(y_shift, x_shift), order=3)
    masked_fusion_img = fusion_img_shift
    return masked_fusion_img


print("norm fusion shape =", norm_fusion.shape)
print("norm mrs shape =", norm_mrs.shape)
fct_chi2 = lambda x : chi2(norm_fusion, norm_mrs, angle=x[0], y_shift=x[1], x_shift=x[2])
result = scipy.optimize.minimize(
                        fct_chi2, 
                        (0, 0, 0), 
                        method='Nelder-Mead', 
                        options={'disp': True, 'maxiter': 2000, 'maxfev': 5000, 'xatol':1e-3, 'fatol':1e-3},
                            )

print(result)



img = get_result_image(norm_fusion, *result.x)

img_final = np.ascontiguousarray(img.astype(np.float64))
mrs_padded_final = np.ascontiguousarray(mrs_padded.astype(np.float64))

interactive_align(mrs_padded_final, img_final, np.ones_like(img))
plt.imshow(img)
plt.show()
