""" 
Script to resample and select FoV of MIRIM image. Then select the best rotation and shift to align MRS and MIRIM images.
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from skimage.transform import rescale, rotate
from scipy.ndimage import shift
from scipy.signal import correlate2d

from surfh.Preprocessing import model_creation


from matplotlib.widgets import Slider, Button
from skimage.transform import rotate
from scipy.ndimage import shift

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

# Exemple d’utilisation :
# interactive_align(mirim_rescaled, adj)


def rotate_shift(mirim_rescaled, angle, y_shift, x_shift):
    mirim_img_rot = rotate(mirim_rescaled, angle=angle, resize=False, order=3, preserve_range=True)
    mirim_img_shift = shift(mirim_img_rot, shift=(y_shift, x_shift), order=3)
    return mirim_img_shift

def apply_mask(cube, mask, wavelength):
    """Apply a mask to the data cube.

    Parameters
    ----------
    mask : np.ndarray
        A boolean array with the same spatial dimensions as the data cube.
        True values indicate valid data points, while False values indicate masked points.
    """
    ch_limit = [5.66, 6.53, 7.51, 8.67, 10.02, 11.55, 13.34, 15.41, 17.70, 20.69, 24.19]
    wavelengths = wavelength
    # Apply the mask to each wavelength slice
    w_start = w_stop = 0
    for i in range(mask.shape[0]):
        w_stop = np.where(wavelengths < ch_limit[i])[0][-1] +1
        slice_mask = slice(w_start, w_stop)
        cube[slice_mask,:,:] = cube[slice_mask,:,:]*mask[i]
        w_start = w_stop
    return cube

def normalize(image):
    # return (image - np.mean(image)) / np.std(image)
    return image / np.max(image)

def match_region(big_image, small_image):
    big_norm = normalize(big_image)
    small_norm = normalize(small_image)

    # cross-correlation
    corr = correlate2d(big_norm, small_norm, mode='valid')

    # trouver l'indice du maximum
    y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)

    # extraire la zone correspondante
    H_s, W_s = small_image.shape
    matched_region = big_image[y_max:y_max+H_s, x_max:x_max+W_s]

    return matched_region, (y_max, x_max)

def main():
    dir = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'
    miri_raw_dir = dir + 'Raw_MIRIM/'
    miri_corrected_dir = dir + 'Corrected_MIRIM/'


    npix = 125
    list_chan = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b']
    imshape = (npix, npix)

    step = 0.1  # arcsec
    paths, step_angle = model_creation.initialize_parameters(dir, step)

    wavel_axis, templates, sotf = model_creation.load_simulation_data(paths, list_chan)

    data_dict = model_creation.load_data(list_chan, paths["save_filter_corrected_dir"])

    instruments = model_creation.create_instruments(data_dict, list_chan) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape)

    data = list()
    for chan in list_chan:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    maps = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_9_MO_4_Temp_6_nit_500_mu_5.00e+06_SD_True/res_x.npy')
    cube = np.array(MRSModel.mapsToCube(maps))
    masks = MRSModel.make_mask(ndata)
    masks = np.array(masks)
    masked_cube_MRS = apply_mask(cube, masks, wavel_axis)
    pce_miri = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/MIRIM/MIRI.F1280W_1ABC_2ABC_3ABC_4ABC.npy')[0]
    pce_miri = pce_miri[:len(wavel_axis)]
    integrated_cube = np.sum(masked_cube_MRS*pce_miri[..., np.newaxis, np.newaxis], axis=0)

    list_xmax = [199, 198, 198, 199, 199, 200, 201, 199, 199]
    list_ymax = [218, 217, 217, 218, 218, 219, 219, 218, 218]
 
    i = 0
    for file in sorted(os.listdir(miri_raw_dir)):
        with fits.open(miri_raw_dir + file) as hdul:
            mirim_data = hdul[1].data
            mirim_header = hdul[1].header

            res_pix = np.abs(mirim_header['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
            scale_factor = res_pix / 0.1

            mirim_rescaled = rescale(
                mirim_data, 
                scale=scale_factor, 
                order=3,              # interpolation bicubique
                preserve_range=True,  # garder les valeurs d’intensité
                anti_aliasing=True
            )

            angle, y_shift, x_shift = [-9.036e+01, 1.875e+00, -2.671e+01]
    
            mirim_img_aligned = rotate_shift(mirim_rescaled, angle, y_shift, x_shift)
            mirim_img_aligned = mirim_img_aligned[500:1000, 100:600]

#             fig, ax2 = plt.subplots(1, 2, figsize=(10, 5))
#             im0 = ax2[0].imshow(mirim_rescaled, cmap='viridis')
#             ax2[0].set_title('MIRIM F1280W')
#             cbar0 = plt.colorbar(im0, ax=ax2[0], orientation='vertical')
# # 
#             im1 = ax2[1].imshow(mirim_img_aligned[500:1000, 100:600], cmap='viridis')
#             ax2[1].set_title('Integrated MRS Cube')
#             cbar1 = plt.colorbar(im1, ax=ax2[1], orientation='vertical')
#             cbar1.set_label('Intensité normalisée')
#             plt.show()
# 

            # matched_img, (ymax, xmax)= match_region(mirim_img_aligned, integrated_cube)
            # xmax = xmax +2
            xmax = list_xmax[i]
            ymax = list_ymax[i]
            i += 1
            zoom_miri = mirim_img_aligned[ymax:ymax+integrated_cube.shape[0], xmax:xmax+integrated_cube.shape[1]]
            print(f"Offset for {file} : (y, x) = ({ymax}, {xmax})")

            fig, ax2 = plt.subplots(1, 5, figsize=(10, 5))
            im0 = ax2[0].imshow(mirim_img_aligned, cmap='viridis')
            ax2[0].set_title(f'{file}')
            cbar0 = plt.colorbar(im0, ax=ax2[0], orientation='vertical')

            im1 = ax2[1].imshow(integrated_cube, cmap='viridis')
            ax2[1].set_title('Integrated MRS Cube')
            cbar1 = plt.colorbar(im1, ax=ax2[1], orientation='vertical')
            cbar1.set_label('Intensité normalisée')
            
            im2 = ax2[2].imshow(zoom_miri, cmap='viridis')
            ax2[2].set_title('Zoom MIRIM')
            cbar2 = plt.colorbar(im2, ax=ax2[2], orientation='vertical')
            cbar2.set_label('Intensité normalisée')

            masked_mirim = zoom_miri*masks[6]
            im3 = ax2[3].imshow(masked_mirim, cmap='viridis')
            ax2[3].set_title('masked MIRIM')
            cbar3 = plt.colorbar(im3, ax=ax2[3], orientation='vertical')
            cbar3.set_label('Intensité normalisée')

            norm_mirim = masked_mirim / np.max(masked_mirim)
            norm_mrs = integrated_cube/ np.max(integrated_cube)
            im4 = ax2[4].imshow(norm_mirim-norm_mrs, cmap='viridis')
            ax2[4].set_title('Diff')
            cbar4 = plt.colorbar(im4, ax=ax2[4], orientation='vertical')
            cbar4.set_label('Intensité normalisée')

            # plt.show()
            # interactive_align(zoom_miri, integrated_cube, masks[6])

            hdu = fits.PrimaryHDU(data=zoom_miri)
            header = hdu.header
            header['AUTHOR']   = 'Nicolas Monnier'
            header['NAXIS1']   = zoom_miri.shape[0]   # taille en alpha
            header['NAXIS2']   = zoom_miri.shape[1]   # taille en beta

            header['Rot']    = (-9.036e+01, "Rotation to match MRS surfh image")   # Position Angle (V3) in degrees
            header['Xshift'] = ( -2.671e+01, "First X shift to match MRS surfh image")   
            header['Yshift'] = (  1.875e+00, "First Y shift to match MRS surfh image")   

            header['PixXshift'] = (xmax, "pixel X shift to match MRS surfh image")   
            header['PixYshift'] = (ymax, "pixel Y shift to match MRS surfh image")   

            hdu.writeto(miri_corrected_dir + file, overwrite=True)

    return


if __name__ == "__main__":
    main()