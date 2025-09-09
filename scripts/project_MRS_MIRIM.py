import numpy as np
import os
import matplotlib.pyplot as plt

from rich import print

from surfh.Preprocessing import model_creation
from surfh.ToolsDir import reconstruction
from surfh.Vizualisation import cube_vizualisation
from surfh.ToolsDir.fits_toolbox import save_numpy_to_fits

import click
import logging as log
from astropy.io import fits


import numpy as np
from skimage.transform import warp_polar, rescale, rotate
from skimage.registration import phase_cross_correlation
from skimage.filters import window

import numpy as np
from skimage.feature import match_template

def find_shift(rotated, adj):
    """
    Trouve la position de adj dans rotated.
    Retourne le décalage (y, x) pour aligner rotated avec adj.
    """
    # Calculer la carte de similarité
    result = match_template(rotated, adj)
    
    # Trouver l'indice du meilleur match
    ij = np.unravel_index(np.argmax(result), result.shape)
    shift_y, shift_x = ij  # coordonnées du coin supérieur gauche de adj dans rotated
    
    print(f"Décalage estimé (y, x) = {shift_y}, {shift_x}")
    return shift_y, shift_x



def estimate_rotation_and_scale_via_logpolar(img1, img2):
    """
    img1 : référence (grande). img2 : target (zoomée/rotée).
    Retourne (rot_deg, scale_factor)
    """
    # Appliquer fenêtre pour réduire bord
    w1 = img1 * window('hann', img1.shape)
    w2 = img2 * window('hann', img2.shape)

    # Utiliser le module du spectre
    f1 = np.abs(np.fft.fftshift(np.fft.fft2(w1)))
    f2 = np.abs(np.fft.fftshift(np.fft.fft2(w2)))

    # centre et rayon : prendre le plus petit demi-dim pour être sûr
    center = (f1.shape[0]//2, f1.shape[1]//2)
    radius = min(center)  # rayon max utilisable

    # warp_polar en échelle log
    lp1 = warp_polar(f1, center=center, radius=radius, scaling='log')
    lp2 = warp_polar(f2, center=center, radius=radius, scaling='log')

    # corrélation de phase sur log-polaire
    shift, error, _ = phase_cross_correlation(lp1, lp2, upsample_factor=10)

    # shift[0] correspond aux décalages angulaires (lignes) → rotation
    rot_fraction = shift[0] / lp1.shape[0]   # fraction of full circle
    rot_deg = rot_fraction * 360.0
    # shift[1] correspond au décalage radial (colonnes) → log(scale)
    log_scale = shift[1] / lp1.shape[1]
    scale_factor = np.exp(log_scale)

    return rot_deg, scale_factor, error

import numpy as np
from skimage.transform import rescale, rotate
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

import numpy as np
from skimage.transform import rotate
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

import numpy as np
from skimage.transform import rotate
from skimage.registration import phase_cross_correlation

import numpy as np
from skimage.transform import rotate, warp_polar
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

def align_with_logpolar(mirim_rescaled, adj):
    """
    Aligne mirim_rescaled sur adj en estimant d'abord la rotation via log-polar, 
    puis le décalage via phase_cross_correlation.
    """
    # 1️⃣ Approximer le centre sur mirim_rescaled
    center = (mirim_rescaled.shape[0]//2, mirim_rescaled.shape[1]//2)
    
    # 2️⃣ Fourier magnitude des deux images
    fft_mirim = np.abs(np.fft.fftshift(np.fft.fft2(mirim_rescaled)))
    fft_adj = np.abs(np.fft.fftshift(np.fft.fft2(adj)))
    
    # 3️⃣ Log-polar transform centrée sur approximativement la zone d’intérêt
    lp_mirim = warp_polar(fft_mirim, center=center, scaling='log')
    lp_adj = warp_polar(fft_adj, center=center, scaling='log')
    
    # 4️⃣ Corrélation de phase pour détecter rotation
    shift_lp, error, diffphase = phase_cross_correlation(lp_adj, lp_mirim, upsample_factor=100)
    
    # L'angle en degrés
    angle = (shift_lp[0] / lp_mirim.shape[0]) * 360
    print(f"Angle estimé = {angle:.2f}°")
    
    # 5️⃣ Appliquer rotation
    rotated = rotate(mirim_rescaled, angle, resize=True, order=3, preserve_range=True)
    
    # 6️⃣ Recadrer le centre pour matcher la taille d'adj
    minr = min(rotated.shape[0], adj.shape[0])
    minc = min(rotated.shape[1], adj.shape[1])
    r0 = (rotated.shape[0]-minr)//2
    c0 = (rotated.shape[1]-minc)//2
    rotated_crop = rotated[r0:r0+minr, c0:c0+minc]
    
    # 7️⃣ Corrélation de phase pour estimer le décalage
    shift_est, error, _ = phase_cross_correlation(rotated_crop, adj, upsample_factor=50)
    print(f"Décalage estimé (y, x) = {shift_est}")
    
    # 8️⃣ Appliquer le shift final
    aligned = shift(rotated_crop, shift=-shift_est)
    
    return rotated_crop, angle, shift_est

# Exemple d'utilisation


def find_best_rotation(mirim_rescaled, adj, coarse_step=5, fine_range=2, fine_step=0.2):
    """
    Trouve l'angle optimal pour aligner mirim_rescaled sur adj.
    """
    best_error = np.inf
    best_angle = 0

    # 1️⃣ Recherche coarse
    for angle in range(0, 360, coarse_step):
        rotated = rotate(mirim_rescaled, angle, resize=True, order=3, preserve_range=True)
        
        # crop central pour matcher adj
        minr = min(rotated.shape[0], adj.shape[0])
        minc = min(rotated.shape[1], adj.shape[1])
        r0 = (rotated.shape[0]-minr)//2
        c0 = (rotated.shape[1]-minc)//2
        rot_crop = rotated[r0:r0+minr, c0:c0+minc]
        
        shift_est, error, _ = phase_cross_correlation(rot_crop, adj, upsample_factor=10)
        if error < best_error:
            best_error = error
            best_angle = angle

    print("shift est", shift_est)

    # 2️⃣ Affinage autour du meilleur angle
    angles_fine = np.arange(best_angle-fine_range, best_angle+fine_range+fine_step, fine_step)
    for angle in angles_fine:
        rotated = rotate(mirim_rescaled, angle, resize=True, order=3, preserve_range=True)
        minr = min(rotated.shape[0], adj.shape[0])
        minc = min(rotated.shape[1], adj.shape[1])
        r0 = (rotated.shape[0]-minr)//2
        c0 = (rotated.shape[1]-minc)//2
        rot_crop = rotated[r0:r0+minr, c0:c0+minc]

        shift_est, error, _ = phase_cross_correlation(rot_crop, adj, upsample_factor=50)
        if error < best_error:
            best_error = error
            best_angle = angle

    print(f"Angle optimal ≈ {best_angle:.2f}°")
    return best_angle


def align_images_no_rescale(mirim_rescaled, adj):
    """
    Aligne mirim_rescaled (déjà à la même résolution que adj) sur adj.
    Retourne l'image alignée, l'angle optimal et le décalage (shift).
    """
    best_error = np.inf
    best_angle = 0
    best_shift = (0, 0)

    # 1️⃣ Recherche coarse tous les 5°
    for angle in range(0, 360, 5):
        rotated = rotate(mirim_rescaled, angle, resize=False, order=1, preserve_range=True)

        # centre crop pour matcher la taille d'adj
        minr = min(rotated.shape[0], adj.shape[0])
        minc = min(rotated.shape[1], adj.shape[1])
        def center_crop(img, nr, nc):
            r0 = (img.shape[0]-nr)//2
            c0 = (img.shape[1]-nc)//2
            return img[r0:r0+nr, c0:c0+nc]

        rot_crop = center_crop(rotated, minr, minc)
        adj_crop = center_crop(adj, minr, minc)

        shift_est, error, _ = phase_cross_correlation(rot_crop, adj_crop, upsample_factor=10)
        if error < best_error:
            best_error = error
            best_angle = angle
            best_shift = shift_est

    # 2️⃣ Affinage autour de l'angle coarse
    angles_fine = np.arange(best_angle-2, best_angle+2, 0.2)
    for angle in angles_fine:
        rotated = rotate(mirim_rescaled, angle, resize=False, order=3, preserve_range=True)
        rot_crop = center_crop(rotated, minr, minc)
        shift_est, error, _ = phase_cross_correlation(rot_crop, adj_crop, upsample_factor=50)
        if error < best_error:
            best_error = error
            best_angle = angle
            best_shift = shift_est

    # 3️⃣ Appliquer rotation et shift définitifs
    rotated_final = rotate(mirim_rescaled, best_angle, resize=False, order=3, preserve_range=True)
    aligned = shift(rotated_final, shift=best_shift)

    print(f"Angle optimal = {best_angle:.2f}°")
    print(f"Décalage optimal (y,x) = {best_shift}")

    return aligned, best_angle, best_shift

# Exemple d'utilisation :

# Exemple d'utilisation :
# rot_deg, scale, err = estimate_rotation_and_scale_via_logpolar(mirim_data, adj)
# print("rotation ≈", rot_deg, "deg ; scale ≈", scale)

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.transform import rotate
from scipy.ndimage import shift

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage.transform import rotate
from scipy.ndimage import shift

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage.transform import rotate
from scipy.ndimage import shift

def interactive_align(mirim_rescaled, adj):
    """
    Interface interactive avec sliders (rotation, shift X/Y, alpha)
    et un bouton pour interchanger les images (fond / overlay).
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Mode initial : adj en fond, mirim_rescaled superposé
    base_img = ax.imshow(adj, cmap='viridis', origin='lower')
    overlay_img = ax.imshow(mirim_rescaled, cmap='magma', alpha=0.5, origin='lower')

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

        overlay_img.set_data(shifted)
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
            base_img.set_data(mirim_rescaled)
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

# Exemple d’utilisation :
# interactive_align(mirim_rescaled, adj)

# Exemple d’utilisation :
# interactive_align(mirim_rescaled, adj)



from skimage import transform
from scipy.ndimage import shift
def cross_corr(mirim_img, mrs_img, angle, x_shift, y_shift):
    mirim_img = rotate(mirim_img, angle=angle)
    # mirim_img = np.roll(mirim_img, int(x_shift), axis=0)
    # mirim_img = np.roll(mirim_img, int(y_shift), axis=1)
    # transformer = transform.AffineTransform(rotation=angle, translation=(x_shift, y_shift))
    # mirim_img_transformed = transform.warp(mirim_img, inverse_map=transformer.inverse)
    # plt.imshow(mirim_img_transformed)
    # plt.show()
    mirim_img = shift(mirim_img, (x_shift, y_shift))
    corr = np.sum(mirim_img*mrs_img)
    return (1/corr)*1e15



@click.command()
@click.option('-fd', '--fusion_dir', default='/home/nmonnier/Data/JWST/NGC_7023//Fusion/', type=str, help='Fusion directory')
@click.option('-np', '--npix', default=125, type=int, help='Number of pixels')
@click.option('-hp', '--hyper_parameter', default=1., type=float, help='Hyperparameter value')
@click.option('-ni', '--niter', default=5, type=int, help='Number of iteration.')
@click.option('-nt', '--n_templates', default=4, type=int, help='Number of Templates.')
@click.option('-sd', '--scale_data', default=False, type=bool, help='Scale data from Jy  to Jy/str.')
@click.option('-m', '--method', default='lcg', type=str, help='Method used (default = lcg).')
@click.option('-v', '--verbose', default=False, type=bool, help='Verbose.')
def parse_options(fusion_dir, npix, hyper_parameter, niter, n_templates, scale_data, method, verbose):

    print(f'Options selected are : ') 
    print(f'\t fusion_dir = {fusion_dir}')
    print(f'\t npix = {npix}')
    print(f'\t hyper_parameter = {hyper_parameter}')
    print(f'\t niter = {niter}')
    print(f'\t nTemplates = {n_templates}')
    print(f'\t scale_data = {scale_data}')
    print(f'\t method = {method}')
    print(f'\t verbose = {verbose}')

    # if verbose:
    #     log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    list_chan = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b']
    imshape = (npix, npix)

    # log.info('Initialize basic path parameters')
    step = 0.1  # arcsec
    paths, step_angle = model_creation.initialize_parameters(fusion_dir, step)


    log.info('Load simulation data')
    wavel_axis, templates, sotf = model_creation.load_simulation_data(paths, list_chan)


    log.info('Load MRS data')
    data_dict = model_creation.load_data(list_chan, paths["save_filter_corrected_dir"])


    log.info('Cerate intruments and spectro models')
    instruments = model_creation.create_instruments(data_dict, list_chan) # Warning Here : Rotation is set as -rotation_angle
    MRSModel = model_creation.create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape)

    data = list()
    for chan in list_chan:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)

    # if scale_data:
    #     log.info('Data scaling enable')
    #     ndata = MRSModel.real_data_janskySR_to_jansky(ndata)

    from astropy.wcs import WCS
    from astropy.visualization import simple_norm
    from astropy import units as u
    hdul_mirim = fits.open('/home/nmonnier/Data/JWST/NGC_7023/Fusion/MIRIM/Level3_F1280W_i2d.fits')
    # print(hdul_mirim.info())
    # print(hdul_mirim[1].header)
    mirim_data = hdul_mirim[1].data
    hdr = hdul_mirim[1].header
    wcs = WCS(hdr)

    res_pix = np.abs(hdr['CDELT1'])*u.deg.to(u.arcsec)  # arcsec/pixel
    print(f"Resolution pixel is {res_pix} arcsec/pixel")

    from skimage.transform import rescale
    scale_factor = res_pix / 0.1  # = 1.1

    mirim_rescaled = rescale(
        mirim_data, 
        scale=scale_factor, 
        order=3,              # interpolation bicubique
        preserve_range=True,  # garder les valeurs d’intensité
        anti_aliasing=True
    )
    print(f"mirim shape {mirim_data.shape} -> {mirim_rescaled.shape} with scale factor {scale_factor}")
    
    mirim_rescaled_zoom = mirim_rescaled[800:925,680:805]

    norm = simple_norm(mirim_data, 'sqrt', percent=99.5)


    ndith = [0,1,2,3]  # Dithers to use for the test
    chan_idx = 1  # Channel index to test
    slice_idx = -1  # Last slice index to test
    # _,_,slice_0 = MRSModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)
    # chan_idx = 2  # Channel index to test
    # slice_idx = -1  # Last slice index to test
    # _,_,slice_1 = MRSModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)


    # adj0 = MRSModel.adjoint(ndata)
    maps = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_9_MO_4_Temp_6_nit_500_mu_5.00e+06_SD_True/res_x.npy')
    cube = np.array(MRSModel.mapsToCube(maps))
    masks = MRSModel.make_mask(ndata)
    masks = np.array(masks)
    print(masks.shape)
    adj0 = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Results/lcg_MC_9_MO_4_Temp_6_nit_500_mu_5.00e+06_SD_True/res_cube.npy')

    hdul_mrs = fits.open('/home/nmonnier/Projects/JWST/hyper-vista/simulation/res_CubeMask.fits') 
    cube_MRS = hdul_mrs[0].data
    print(cube.shape, cube_MRS.shape)
    mask_MRS = hdul_mrs['MASKS'].data
    print(mask_MRS.shape)
    wavelength = hdul_mrs['WCS-TABLE'].data['wavelength'][0].squeeze()
    masked_cube_MRS = apply_mask(cube, masks, wavelength)

    pce_miri = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/MIRIM/MIRI.F1280W_1ABC_2ABC_3ABC_4ABC.npy')[0]
    pce_miri = pce_miri[:len(wavelength)]
    integrated_cube = np.sum(masked_cube_MRS*pce_miri[..., np.newaxis, np.newaxis], axis=0)
    print(f"Shape intergrated cube = {integrated_cube.shape}")
    # plt.figure()
    # plt.imshow(integrated_cube)
    # plt.colorbar()
    
    from scipy.ndimage import rotate
# 
#     metadata = {'PA_V3': data_dict['PA_V3']['1c'], 
#                 'TARG_RA': data_dict['target']['1c'][0], 'TARG_DEC':data_dict['target']['1c'][1], 
#                 'RA_V1': data_dict['targetV1']['1c'][0], 'DEC_V1': data_dict['targetV1']['1c'][1], 
#                 'RA_REF': data_dict['targetREF']['1c'][0], 'DEC_REF': data_dict['targetREF']['1c'][1],
#                 'ALPHA_AXIS':MRSModel.alpha_axis, 'BETA_AXIS':MRSModel.beta_axis, 'WAVELENGTH':0}
# # 

    adj = adj0[100]
    # adj = rotate(np.flipud(np.fliplr(adj0[5])), angle=metadata['PA_V3']-360-8.2, reshape=False)
    # fw = MRSModel.forward(adj0)
    # adj1 = MRSModel.adjoint(fw)


    # import numpy as np
    # from skimage.transform import rotate
    # from skimage.registration import phase_cross_correlation
    # from scipy.ndimage import zoom

    # # On réduit la taille de adj pour matcher mirim_data (au moins grossièrement)
    # zoom_factor = mirim_rescaled.shape[0] / adj.shape[0]
    # adj_resized = zoom(adj, zoom_factor)
    # plt.figure()
    # plt.imshow(mirim_rescaled_zoom)
    # plt.colorbar()

    # interactive_align(mirim_rescaled_zoom, integrated_cube)
    # plt.show()
    print(mirim_rescaled_zoom.shape)


    import scipy
    res1 = cross_corr(mirim_rescaled_zoom, integrated_cube, 0,0,0)
    res2 = cross_corr(mirim_rescaled_zoom, integrated_cube, -93*np.pi/180 ,-26, 2)
    res3 = cross_corr(mirim_rescaled_zoom, integrated_cube, -93*np.pi/180 , 2, -26)
    print(f"Res1 = {res1}, Res2 = {res2}, Res3 = {res3}")

    fct_to_minimize = lambda x : cross_corr(mirim_rescaled_zoom, integrated_cube, *x)
    np.save('/home/nmonnier/mirim_img.npy', mirim_rescaled_zoom)
    np.save('/home/nmonnier/mrs_img.npy', integrated_cube)
    result = scipy.optimize.minimize(fct_to_minimize, (0, 0, 0), method='BFGS', options={'disp':True})

    print(result)


    
    # rot_deg, scale_factor, error = estimate_rotation_and_scale_via_logpolar(mirim_rescaled_zoom, integrated_cube)
    # print(f"Estimated rotation: {rot_deg:.2f} deg, scale factor: {scale_factor:.4f}, error: {error:.4e}")
    # mirim_rescaled_zoom_rotated = rotate(mirim_rescaled_zoom, angle=-rot_deg, reshape=False)

    # _, angle, shift_pix = align_images_no_rescale(mirim_rescaled_zoom, integrated_cube)
    # print(f"Final alignment angle: {angle:.2f} deg, shift: {shift_pix}")

    # best_angle = find_best_rotation(mirim_rescaled_zoom, integrated_cube)
    # print(f"Best rotation angle found: {best_angle:.2f} deg")
    

    # rot_image, best_angle, best_shift = align_with_logpolar(mirim_rescaled_zoom, integrated_cube)
    # print(f"Align with log-polar: angle {best_angle:.2f} deg, shift {best_shift}")

    # shift_y, shift_x = find_shift(rot_image, integrated_cube)
    # aligned = rot_image[shift_y:shift_y, shift_x:shift_x]


    # fig, ax = plt.subplots(1, 6, figsize=(12, 6))


    # pce = np.load('/home/nmonnier/Data/JWST/Simulation/Orion/PCE/pce.npy')
    # pce_1280 = pce[4
    # im0 = ax[0].imshow(mirim_rescaled, cmap='viridis', norm=norm)
    # ax[0].set_title('MIRIM F1280W')
    # # Forcer les axes en degrés décimaux
    # ax[0].set_ylabel("Déclinaison (deg)")
    # cbar0 = plt.colorbar(im0, ax=ax[0], orientation='vertical')
    # cbar0.set_label(hdul_mirim[1].header.get('BUNIT', 'Intensité'))    
    # ax[1].imshow(mirim_rescaled_zoom, cmap='viridis')
    # ax[1].set_title('Zoom')
    # ax[2].imshow(adj, cmap='viridis')
    # ax[2].set_title('Adjoint MRS Channel 1B Slice')
    # ax[3].imshow(mirim_rescaled_zoom_rotated, cmap='viridis')
    # ax[3].set_title(f'Zoom rot {rot_deg:.1f} deg')
    # ax[4].imshow(rot_image, cmap='viridis')
    # ax[4].set_title(f'Aligné (shift {shift_pix[0]:.1f}, {shift_pix[1]:.1f})')
    # ax[5].imshow(aligned, cmap='viridis')
    # ax[5].set_title('Décalé')
    # plt.show()


    # alpha_coord, beta_coord = model_creation.get_axis(MRSModel)
    # extent = [alpha_coord.min(), alpha_coord.max(), beta_coord.min(), beta_coord.max()]
    # plt.figure()
    # plt.imshow(slice_0, cmap='viridis', alpha=0.5)
    # plt.imshow(slice_1, cmap='viridis', extent=extent, alpha=0.5)
    # plt.colorbar()

    # cube_vizualisation.plot_cube(adj0, np.arange(adj0.shape[0]))
    # MRSModel.project_FOV()   
    # plt.show()    




if __name__ == "__main__":
    parse_options()