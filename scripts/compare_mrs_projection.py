import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from matplotlib.patches import Polygon



import numpy as np
import os
import udft
from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt

from rich import print
from rich.progress import track
from rich.console import Console

from astropy import units as u
from astropy.coordinates import Angle
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs, realmiri, instru
from surfh.Models import spectroModel
from surfh.Vizualisation import slices_vizualisation, cube_vizualisation
from surfh.Simulation import fusion_CT
from aljabr import LinOp, dottest
from scipy import ndimage
from surfh.Simulation.fusion_CT import QuadCriterion_MRS
import argparse
import click
import itertools

import logging as log

def create_model(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, data_dict):
    """Create the spectrograph model."""
    main_pointing = instru.Coord(0, 0)

    pointings = []

    for idx, chan in enumerate(instruments.keys()):
        pointing_chan = [main_pointing + instru.Coord(RA, DEC) for RA, DEC in data_dict['target'][chan]]
        # pointing_chan = [main_pointing + instru.Coord(ra[idx], dec[idx]) for idx in range(len(ra))]
        pointings.append(instru.CoordList(pointing_chan).pix(step_angle))
        print("pointing_chan = ", pointing_chan)

    # alpha_axis = origin_alpha_axis + data_dict['target']['2a'][2][0]
    # beta_axis = origin_beta_axis + data_dict['target']['2a'][2][1]
    mean_alpha = np.mean([data_dict['target']['1a'][dith][0] for dith in range(4)])
    mean_beta = np.mean([data_dict['target']['1a'][dith][1] for dith in range(4)])
    # mean_alpha = data_dict['target']['1a'][0][0] 
    # mean_beta = data_dict['target']['1a'][0][1] 
    print("mean_alpha = ", mean_alpha)
    print("mean_beta = ", mean_beta)
    alpha_axis = origin_alpha_axis + mean_alpha
    beta_axis = origin_beta_axis + mean_beta
    print("alpha_axis = ", alpha_axis)
    print("beta_axis = ", beta_axis)
    # alpha_axis = origin_alpha_axis + ra[0]#360-44.618321664549896#mean_alpha
#     # beta_axis = origin_beta_axis + dec[0]#68.17383920898915#mean_beta


    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings), alpha_axis, beta_axis


def create_instruments(data_dict, list_chan):
    """Create instrument configurations for each channel."""
    instruments = {}

    channel_specs = {
        '1a': (21, 3320, 3710, 0.196, 3.2/3600, 3.7/3600),
        '1b': (21, 3190, 3750, 0.196, 3.2/3600, 3.7/3600),
        '1c': (21, 3100, 3610, 0.196, 3.2/3600, 3.7/3600),
        '2a': (17, 2990, 3110, 0.196, 4.0/3600, 4.8/3600),
        '2b': (17, 2750, 3170, 0.196, 4.0/3600, 4.8/3600),
        '2c': (17, 2860, 3300, 0.196, 4.0/3600, 4.8/3600),
        '3a': (16, 2530, 2880, 0.245, 5.2/3600, 6.2/3600),
        '3b': (16, 1790, 2640, 0.245, 5.2/3600, 6.2/3600),
        '3c': (16, 1980, 2790, 0.245, 5.2/3600, 6.2/3600),
        '4a': (12, 1460, 1930, 0.273, 6.6/3600, 7.7/3600),
        '4b': (12, 1680, 1760, 0.273, 6.6/3600, 7.7/3600),
        '4c': (12, 1630, 1330, 0.273, 6.6/3600, 7.7/3600)
    }

    for chan, (n_slit, r_min, r_max, det_pix_size, fov_x, fov_y) in channel_specs.items():
        if chan in list_chan:
            print(f"Rotation {chan} = {data_dict['rotation'][chan]}")
            spec_blur = instru.SpectralBlur(np.mean([r_min, r_max]))
            instruments[chan] = instru.IFU(
                fov=instru.FOV(fov_x, fov_y, origin=instru.Coord(0, 0), angle=-data_dict['rotation'][chan]),
                det_pix_size=det_pix_size,
                n_slit=n_slit,
                w_blur=spec_blur,
                pce=None,
                wavel_axis=wavelength_mrs.get_mrs_wavelength(chan),
                name=chan.upper()
            )

    return instruments


def load_data(list_chan, save_filter_corrected_dir):
    """Load data for the specified channels."""
    data_dict = {'data': {}, 'target': {}, 'rotation': {}}

    datashape = {
        '1a': (21, 1050, 19), '1b': (21, 1213, 19), '1c': (21, 1400, 19),
        '2a': (17, 970, 24), '2b': (17, 1124, 24), '2c': (17, 1300, 24),
        '3a': (16, 769, 24), '3b': (16, 892, 24), '3c': (16, 1028, 24),
        '4a': (12, 542, 27), '4b': (12, 632, 27), '4c': (12, 717, 27)
    }
    for chan in list_chan:
        data_dict['data'][chan] = []
        data_dict['target'][chan] = []
        data_dict['rotation'][chan] = 0.
    i=0

    coords = []
    for file in sorted(os.listdir(save_filter_corrected_dir)):
        for chan in list_chan:
            if 'ch1a' in file:
                with fits.open(os.path.join(save_filter_corrected_dir, file)) as hdul:
                    header = hdul[0].header
                    TARG_RA = header['TARG_RA']     
                    TARG_DEC = header['TARG_DEC']       
                    coords.append((TARG_RA, TARG_DEC))      

    for file in sorted(os.listdir(save_filter_corrected_dir)):
        for chan in list_chan:
            if chan in file:
                with fits.open(os.path.join(save_filter_corrected_dir, file)) as hdul:
                    header = hdul[0].header
                    PA_V3 = header['PA_V3']
                    TARG_RA = header['TARG_RA']
                    TARG_DEC = header['TARG_DEC']
                    data_shape = (header['NSlits'], header['NWavel'], header['Nalpha'])
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1, 0, 2)

                    data_dict['data'][chan].append(ndata)
                    data_dict['target'][chan].append((TARG_RA, TARG_DEC))
                    print(TARG_RA, TARG_DEC)
                    # data_dict['target'][chan].append((permutations[idx][i%4][0], permutations[idx][i%4][1]))
                    data_dict['rotation'][chan] = PA_V3
                    i += 1

    return data_dict


def load_simulation_data(paths, step, step_angle, Npix):
    """Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = (np.arange(imshape[0]) * step_angle - np.mean(np.arange(imshape[0]) * step_angle))
    origin_beta_axis = np.arange(imshape[1]) * step_angle - np.mean(np.arange(imshape[1]) * step_angle)
    
    
    wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_SS4.npy'))
    templates = None
    sotf = None
    
    return origin_alpha_axis, origin_beta_axis, wavel_axis, templates, sotf


def initialize_parameters(fusion_dir_path, step=0.1):
    """Initialize global parameters."""
    paths = {
        'psf_dir': os.path.join(fusion_dir_path, 'PSF/'),
        'template_dir': os.path.join(fusion_dir_path, 'Templates/'),
        'save_filter_corrected_dir': os.path.join(fusion_dir_path, 'Filtered_slices/'),
        'result_path': os.path.join(fusion_dir_path, 'Results/'),
        'mask_path': os.path.join(fusion_dir_path, 'Masks/'),
        'pce_path': os.path.join(fusion_dir_path, 'PCE/')
    }
    step_angle = Angle(step, u.arcsec).degree

    return paths, step_angle


def parse_options():

    fusion_dir = "/home/nmonnier/Data/JWST/small_NGC/Fusion/"
    npix = 125
    
    list_chan = ['1a','2a']

    step = 0.1  # arcsec
    paths, step_angle = initialize_parameters(fusion_dir, step)

    origin_alpha_axis, origin_beta_axis, wavel_axis, templates, sotf = load_simulation_data(paths, step, step_angle, npix)

    data_dict = load_data(list_chan, paths["save_filter_corrected_dir"])

    instruments = create_instruments(data_dict, list_chan)
    spectroModel,alpha_coord, beta_coord = create_model(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, data_dict)

    data = list()
    for chan in list_chan:
        data.append(np.array(data_dict['data'][chan]).ravel())
    ndata = np.concatenate(data)


    not_weigthed = spectroModel.test_project_mrs_slice(ndata, 1, -1)

    return not_weigthed, alpha_coord, beta_coord


def reproject_simulated_image_with_coords(not_weigthed, alpha_coord, beta_coord, ref_fits_path):
    """
    Reprojette une image simulée sur le WCS d’un fichier FITS en utilisant alpha/beta (en arcsec).

    Parameters:
        not_weigthed (2D np.array): image simulée (beta x alpha)
        alpha_coord (1D np.array): coordonnées RA relatives (en arcsec)
        beta_coord (1D np.array): coordonnées Dec relatives (en arcsec)
        ref_fits_path (str): chemin vers un fichier FITS de référence (pour WCS)

    Returns:
        reproj_data (2D np.array): image reprojetée sur le WCS FITS
        ref_wcs (astropy.wcs.WCS): WCS cible
    """

    from astropy.wcs import WCS
    from reproject import reproject_interp
    import numpy as np
    from astropy.io import fits

    # 1. WCS de référence
    with fits.open(ref_fits_path) as ref_hdul:
        ref_wcs = WCS(ref_hdul[1].header, ref_hdul)
        ref_shape = (ref_hdul[1].header["NAXIS2"], ref_hdul[1].header["NAXIS1"])
        ref_crpix1 = ref_hdul[1].header["CRPIX1"]
        ref_crpix2 = ref_hdul[1].header["CRPIX2"]
        ref_ra, ref_dec, _ = ref_wcs.all_pix2world(ref_crpix1, ref_crpix2, 0, 0)


    # 2. Construction d’un WCS local basé sur alpha/beta
    sim_wcs = WCS(naxis=2)
    sim_wcs.wcs.crval = [ref_ra, ref_dec]  # centre absolu (RA, Dec en degrés)
    sim_wcs.wcs.cdelt = [np.diff(alpha_coord).mean() / 3600.0, np.diff(beta_coord).mean() / 3600.0]  # deg/pix
    sim_wcs.wcs.crpix = [len(alpha_coord) // 2, len(beta_coord) // 2]  # centre image
    sim_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # 3. Reprojection
    data_3d = not_weigthed[np.newaxis, :, :]
    reproj_data, _ = reproject_interp((data_3d, sim_wcs), ref_wcs, shape_out=(1,) + ref_shape)
    reproj_data = reproj_data[0]

    return reproj_data, ref_wcs


from astropy.wcs import WCS
from astropy.io.fits import Header

def build_wcs_from_coords(alpha, beta):
    # Supposé uniformément échantillonné
    naxis1 = len(alpha)
    naxis2 = len(beta)

    cdelt_alpha = (alpha[-1] - alpha[0]) / (naxis1 - 1)
    cdelt_beta = (beta[-1] - beta[0]) / (naxis2 - 1)

    crval1 = alpha[0]
    crval2 = beta[0]
    crpix1 = 1
    crpix2 = 1

    header = Header()
    header['NAXIS'] = 2
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = crval1
    header['CRVAL2'] = crval2
    header['CRPIX1'] = crpix1
    header['CRPIX2'] = crpix2
    header['CDELT1'] = cdelt_alpha
    header['CDELT2'] = cdelt_beta
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'

    return WCS(header)

from astropy.wcs import WCS
from astropy.io.fits import Header
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
import numpy as np


from astropy.io import fits
from astropy.wcs import WCS

def extract_wcs_2d_from_header(fits_file):
    with fits.open(fits_file) as hdul:
        header = hdul[1].header.copy()  # <--- HDU[1] pour rester cohérent
        
        # Supprimer les mots-clés du 3e axe
        for key in list(header.keys()):
            if "3" in key:
                del header[key]

        # Forcer 2D
        header["NAXIS"] = 2
        header.pop("NAXIS3", None)
        header.pop("WCSAXES", None)

        # Forcer les axes célestes
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"
        
        wcs_2d = WCS(header)
    
    return wcs_2d


def reproject_simulated_image_with_coords(alpha_coord, beta_coord, image_2d, ref_fits_filename, ref_shape):
    """
    Reprojette une image simulée (image_2d) définie sur un plan (alpha, beta) en degrés
    vers le plan céleste défini par ref_wcs_3d (WCS 3D JWST) projeté en 2D.

    Parameters
    ----------
    alpha_coord : 1D np.array
        Coordonnées RA-like (en degrés), axe X de l'image simulée
    beta_coord : 1D np.array
        Coordonnées Dec-like (en degrés), axe Y de l'image simulée
    image_2d : 2D np.array
        Image à reprojeter, shape = (len(beta_coord), len(alpha_coord))
    ref_wcs_3d : astropy.wcs.WCS
        WCS de référence (3D)
    ref_shape : tuple
        Dimensions 2D de la référence (NAXIS2, NAXIS1)

    Returns
    -------
    reproj_data : 2D np.array
        Image reprojetée dans le plan du ref_wcs
    ref_wcs_2d : astropy.wcs.WCS
        WCS 2D utilisé comme référence
    """
    # 1. Extraire le WCS 2D (plan RA-Dec) de la WCS 3D
    # ref_wcs_2d = ref_wcs_3d.slice((slice(None), slice(None), 0))  # Retirer l'axe spectral

    ref_wcs_2d = extract_wcs_2d_from_header(ref_fits_filename)

    ref_shape_2d = (ref_shape[0], ref_shape[1])

    # 2. Construire le WCS 2D pour l'image simulée
    naxis1 = len(alpha_coord)
    naxis2 = len(beta_coord)

    # Calcul des pas (supposés réguliers)
    cdelt1 = (alpha_coord[-1] - alpha_coord[0]) / (naxis1 - 1)
    cdelt2 = (beta_coord[-1] - beta_coord[0]) / (naxis2 - 1)

    header = Header()
    header['NAXIS'] = 2
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CRVAL1'] = alpha_coord[0]
    header['CRVAL2'] = beta_coord[0]
    header['CRPIX1'] = 1
    header['CRPIX2'] = 1
    header['CDELT1'] = cdelt1
    header['CDELT2'] = cdelt2

    sim_wcs_2d = WCS(header)

    from astropy.wcs.utils import celestial_frame_to_wcs

    print("Sim WCS celestial axes:", sim_wcs_2d.has_celestial)
    print("Ref WCS celestial axes:", ref_wcs_2d.has_celestial)

    print("Sim WCS naxis:", sim_wcs_2d.naxis)
    print("Ref WCS naxis:", ref_wcs_2d.naxis)
    print("Image shape:", image_2d.shape)
    print("Ref shape out:", ref_shape_2d)

    ###########
    ## DEBUG ##
    ###########

    # Debug coverage: coord bounds
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.coordinates import SkyCoord
    import astropy.units as u   

    # Image simulée (entrée)
    alpha_min, alpha_max = np.min(alpha_coord), np.max(alpha_coord)
    beta_min, beta_max = np.min(beta_coord), np.max(beta_coord)
    print(f"Simulated alpha range: {alpha_min:.6f} -> {alpha_max:.6f} deg")
    print(f"Simulated beta  range: {beta_min:.6f} -> {beta_max:.6f} deg")

    # Coord centrale
    center_sim = SkyCoord(
        ra=(alpha_min + alpha_max)/2 * u.deg,
        dec=(beta_min + beta_max)/2 * u.deg,
        frame='icrs'
    )

    # Position dans le WCS de référence
    try:
        px, py = ref_wcs_2d.world_to_pixel(center_sim)
        print(f"Simulated image center projects to reference WCS pixel: ({px:.1f}, {py:.1f})")
        print(f"Reference shape: {ref_shape_2d}")
        if (0 <= px < ref_shape_2d[1]) and (0 <= py < ref_shape_2d[0]):
            print("✅ Simulated image is at least partially within the reference field of view.")
        else:
            print("⚠️ Simulated image lies completely outside the reference field.")
    except Exception as e:
        print("🚫 Error converting simulated coords to ref WCS:", e)

    import matplotlib.pyplot as plt

    # Coins de l'image simulée
    corners_sim = SkyCoord([
        [alpha_min, beta_min],
        [alpha_min, beta_max],
        [alpha_max, beta_min],
        [alpha_max, beta_max]
    ] * u.deg, frame='icrs')

    # Convertir dans le plan pixel du WCS de référence
    pix_coords = ref_wcs_2d.world_to_pixel(corners_sim)

    plt.figure()
    plt.title("Position des coins de l'image simulée dans le WCS de référence")
    plt.imshow(np.zeros(ref_shape_2d), origin='lower')  # Fond vide, même taille
    plt.plot(pix_coords[0], pix_coords[1], 'ro', label='Sim image corners')
    plt.xlim(0, ref_shape_2d[1])
    plt.ylim(0, ref_shape_2d[0])
    plt.legend()
    plt.grid(True)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.show()





    # 3. Reprojection
    reproj_data, _ = reproject_interp(
        (image_2d, sim_wcs_2d),
        ref_wcs_2d,
        shape_out=ref_shape_2d
    )

    return reproj_data, ref_wcs_2d


# Get all FITS files

dir = "/home/nmonnier/Data/JWST/NGC_7023/"
fits_files = [
    dir + "jw01192001001_0310v_00001_mirifulong_s3d.fits",
    dir + "jw01192001001_0310v_00002_mirifulong_s3d.fits",
    dir + "jw01192001001_0310v_00003_mirifulong_s3d.fits",
    dir + "jw01192001001_0310v_00004_mirifulong_s3d.fits",
]

# Select a reference file (first FITS file)
ref_file = fits_files[0]
with fits.open(ref_file) as ref_hdu:
    ref_wcs = WCS(ref_hdu[1].header, ref_hdu)  # Pass HDUList to handle -TAB coordinates
    ref_shape = (ref_hdu[1].header["NAXIS2"], ref_hdu[1].header["NAXIS1"])  # 2D shape


# Gèle la coordonnée spectrale (lambda) pour ne garder que RA/Dec
ref_shape_2d = (ref_shape[0], ref_shape[1])  # (NAXIS2, NAXIS1)


not_weigthed, alpha_coord, beta_coord = parse_options()  # Load the simulated data and create the spectrograph model

reproj_data, ref_wcs_2d = reproject_simulated_image_with_coords(
    alpha_coord, beta_coord, not_weigthed, ref_file, ref_shape
)

plt.figure(figsize=(10, 10))
ax = plt.subplot(projection=ref_wcs_2d)
ax.imshow(reproj_data, origin='lower', cmap='plasma')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
plt.grid(color='white', ls='dotted')
plt.title("Simulated image reprojected onto JWST WCS")
plt.show()
