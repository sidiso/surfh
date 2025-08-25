
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
from surfh.Models import wavelength_mrs, realmiri, instru, metadataMRS
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

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp
from matplotlib.patches import Polygon


def create_model(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, data_dict):
    """Create the spectrograph model."""
    main_pointing = instru.Coord(0, 0)

    pointings = []

    print("Creating pointings for each channel...")
    print(data_dict['target'])

    # Delta offsets for each channel based on the pointing reference ch1
    delta_pointing = [(0,0),
                      (0.087/3600, 0.288/3600),
                      (0.654/3600, 0.002/3600),
                      (-0.644/3600, -0.261/3600)]
    
    # Pointing correction for SN2023fyq
    pointing_correction = {'1a' : (0, 0),  '1b' : (-1.00E-07, -2.70E-05),  '1c' : (2.72E-05, 5.48E-05),  # ch1
                           '2a' : (-2.81E-05, 2.84E-05), '2b' : (-2.78E-05, -5.28E-05), '2c' : (3.47E-05, -4.86E-05),  # ch2
                           '3a' : (-1.20E-06, 1.20E-06), '3b' : (-1.90E-06, -6.73E-05), '3c' : (-1.00E-06,-5.60E-05),  # ch3
                           '4a' : (-1.07E-04, -2.91E-04), '4b' : (-1.10E-04,-3.23E-04), '4c' : (-1.40E-04,-2.86E-04)} # ch4

    RA_REF  = data_dict['targetREF']['1a'][0]  # Reference RA for pointing
    DEC_REF = data_dict['targetREF']['1a'][1]  # Reference DEC for pointing
    for idx, chan in enumerate(instruments.keys()):
        # RA_CENTER = RA_REF  - data_dict['targetREF'][chan][0]
        # DEC_CENTER = DEC_REF - data_dict['targetREF'][chan][1]
# # 
#         DETLA_RA, DELTA_DEC = metadataMRS.get_chan_delta_pointing(chan)
#         RA_CORR, DEC_CORR = metadataMRS.get_pointing_correction_SN2023fyq(chan)
#         # RA - DITH_RA seems so be the working solution here
#         pointing_chan = [main_pointing + instru.Coord(-RA_CORR - DITH_RA + DETLA_RA, -DEC_CORR + DITH_DEC + DELTA_DEC) for (RA, DEC), (DITH_RA, DITH_DEC) in zip(data_dict['target'][chan], data_dict['dither'][chan])]


        RA_CORR, DEC_CORR = metadataMRS.get_band_delta_pointing(chan)
        pointing_chan = [main_pointing + instru.Coord( -RA_CORR - DITH_RA, -DEC_CORR + DITH_DEC) for (RA, DEC), (DITH_RA, DITH_DEC) in zip(data_dict['target'][chan], data_dict['dither'][chan])]
        # pointing_chan = [main_pointing + instru.Coord(ra[idx], dec[idx]) for idx in range(len(ra))]
        pointings.append(instru.CoordList(pointing_chan).pix(step_angle))
        print("pointing_chan = ", pointing_chan)
        print("pointing chan pix = ", pointings[-1])

    # alpha_axis = origin_alpha_axis + data_dict['target']['2a'][2][0]
    # beta_axis = origin_beta_axis + data_dict['target']['2a'][2][1]
    mean_alpha = np.mean([data_dict['target']['1a'][dith][0] for dith in range(4)])
    mean_beta = np.mean([data_dict['target']['1a'][dith][1] for dith in range(4)])
    mean_alpha = 0#data_dict['target']['1a'][0][0] 
    mean_beta = 0#data_dict['target']['1a'][0][1] 
    print("mean_alpha = ", mean_alpha)
    print("mean_beta = ", mean_beta)
    alpha_axis = origin_alpha_axis + mean_alpha
    # alpha_axis = np.flip(alpha_axis)  # Flip the axis to match the expected orientation

    beta_axis = origin_beta_axis + mean_beta
    
    # alpha_axis = origin_alpha_axis + ra[0]#360-44.618321664549896#mean_alpha
#     # beta_axis = origin_beta_axis + dec[0]#68.17383920898915#mean_beta

    # Dither 001
    # alpha_axis = origin_alpha_axis + 315.3822205899458 - 360 #+ (-0.07481027889940606/3600)
    # beta_axis = origin_beta_axis + 68.17373041466647 #+ (1.2372487603595346/3600)
    
    # Dither 002
    # alpha_axis = origin_alpha_axis + 186.44099109978166 - 360 #+ (-0.07481027889940606/3600)
    # beta_axis = origin_beta_axis + 12.663165115171752 #+ (1.2372487603595346/3600)
    print("mean alpha_axis = ", np.mean(alpha_axis))
    print("mean beta_axis = ", np.mean(beta_axis))

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
    data_dict = {'data': {}, 'target': {}, 'targetV1' :{}, 'targetREF': {}, 'dither': {}, 'rotation': {}}

    datashape = {
        '1a': (21, 1050, 19), '1b': (21, 1213, 19), '1c': (21, 1400, 19),
        '2a': (17, 970, 24), '2b': (17, 1124, 24), '2c': (17, 1300, 24),
        '3a': (16, 769, 24), '3b': (16, 892, 24), '3c': (16, 1028, 24),
        '4a': (12, 542, 27), '4b': (12, 632, 27), '4c': (12, 717, 27)
    }
    for chan in list_chan:
        data_dict['data'][chan] = []
        data_dict['target'][chan] = []
        data_dict['targetV1'][chan] = []
        data_dict['targetREF'][chan] = []
        data_dict['dither'][chan] = []
        data_dict['rotation'][chan] = 0.
    i=0

    print("Order of channels loading : ")
    for file in sorted(os.listdir(save_filter_corrected_dir)):
        for chan in list_chan:
            if chan in file:
                with fits.open(os.path.join(save_filter_corrected_dir, file)) as hdul:
                    print(f"Loading data for channel {chan} from file {file}")
                    header = hdul[0].header
                    PA_V3 = header['PA_V3']
                    TARG_RA = header['TARG_RA']  # Adjust RA to match the expected range
                    TARG_DEC = header['TARG_DEC']   # Adjust DEC to match the expected range
                    DITHER_RA = header['XOFFSET']
                    DITHER_DEC = header['YOFFSET']
                    RA_V1 = header['RA_V1']
                    DEC_V1 = header['DEC_V1']
                    RA_REF = header['RA_REF']
                    DEC_REF = header['DEC_REF']
                    data_shape = (header['NSlits'], header['NWavel'], header['Nalpha'])
                    data = hdul[0].data
                    ndata = data.reshape(data_shape[1], data_shape[0], data_shape[2])
                    ndata = ndata.transpose(1, 0, 2)

                    # TODEL debug Targ RA/DEC dither 002
                    # TARG_RA = 186.44099109978166 - 360# + (-0.4932087719687388/3600)
                    # TARG_DEC = 12.663165115171752# + (2.1608617613007937/3600)

                    data_dict['data'][chan].append(ndata)
                    data_dict['target'][chan].append((TARG_RA, TARG_DEC))
                    data_dict['targetV1'][chan]= (RA_V1, DEC_V1)
                    data_dict['targetREF'][chan] = (RA_REF, DEC_REF)
                    data_dict['dither'][chan].append((DITHER_RA, DITHER_DEC))
                    print(TARG_RA, TARG_DEC)
                    # data_dict['target'][chan].append((permutations[idx][i%4][0], permutations[idx][i%4][1]))
                    data_dict['rotation'][chan] = metadataMRS.get_MRS_rotation(chan)
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


def parse_options(ndith=[0,1,2,3], chan_idx=0, slice_idx=-1):

    fusion_dir = "/home/nmonnier/Data/JWST/Point_source/Fusion/"
    # fusion_dir = "/home/nmonnier/Data/JWST/small_NGC/Fusion/"
    npix = 125
    
    list_chan = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c']
    # list_chan = ['1a', '2a']  

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


    print(f"spectroModel pointings = {spectroModel.pointings}")
    numpy_slice, degrid1, degrid2 = spectroModel.test_project_mrs_slice(ndata, chan_idx, slice_idx, ndith=ndith)


    cube = spectroModel.all_slice_to_cube(ndata, ndith=ndith)
    print(f"Caca cube shape = {cube.shape}")
    idx = np.where(np.sum(cube, axis=(1, 2)) != 0)[0]
    print("idx = ", idx)
    n_cube = cube[idx, ...]

    cube_vizualisation.plot_cube(n_cube, np.arange(n_cube.shape[0]))

    return numpy_slice, alpha_coord, beta_coord, data_dict, spectroModel



import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from matplotlib.patches import Polygon

def extract_2d_wcs(wcs_3d, slice_index):
    """
    Extrait un WCS 2D à partir d'un WCS 3D en fixant la dimension spectrale slice_index.
    """
    return wcs_3d.slice([slice_index, slice(None), slice(None)])

def read_fits_slice_and_wcs_2d(fits_file, slice_idx=0):
    with fits.open(fits_file) as hdul:
        wcs_3d = WCS(hdul[1].header, hdul)  # Passe HDUList pour -TAB coord.
        data_cube = hdul[1].data
        data_cube = np.nan_to_num(data_cube, nan=0)
        
        slice_2d = data_cube[slice_idx, :, :]
        wcs_2d = extract_2d_wcs(wcs_3d, slice_idx)
        
    return slice_2d, wcs_2d

def create_custom_wcs_from_coords(alpha_deg, beta_deg):
    """
    Crée un WCS 2D simple en degrés à partir des vecteurs alpha/beta (en degrés).
    alpha_deg et beta_deg doivent être 1D arrays correspondant aux pixels.
    """
    from astropy.wcs import WCS
    from astropy.io import fits
    
    naxis1 = len(alpha_deg)
    naxis2 = len(beta_deg)

    # Création d'un header WCS simple avec pixel scale linéaire
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = naxis1
    header['NAXIS2'] = naxis2
    header['CTYPE1'] = 'RA---TAN'  # Projection tangentielle sur l'ascension
    header['CTYPE2'] = 'DEC--TAN'  # Projection tangentielle sur la déclinaison
    header['CRPIX1'] = naxis1 / 2
    header['CRPIX2'] = naxis2 / 2
    header['CRVAL1'] = alpha_deg[naxis1 // 2]
    header['CRVAL2'] = beta_deg[naxis2 // 2]
    
    # Calcule un pas de pixel moyen
    cdelt1 = (alpha_deg[-1] - alpha_deg[0]) / (naxis1 - 1)
    cdelt2 = (beta_deg[-1] - beta_deg[0]) / (naxis2 - 1)
    header['CDELT1'] = -cdelt1  # RA inversé (souvent négatif)
    header['CDELT2'] = cdelt2
    
    return WCS(header)

def create_custom_wcs(alpha_coord, beta_coord):
    """
    Crée un WCS 2D simple en TAN projection à partir de vecteurs alpha_coord et beta_coord (en degrés).
    alpha_coord et beta_coord sont supposés être des vecteurs 1D (lon, lat) ordonnés.
    """
    naxis1 = len(alpha_coord)
    naxis2 = len(beta_coord)
    
    wcs = WCS(naxis=2)
    
    # Assume alpha_coord et beta_coord sont régulièrement espacés. 
    # Si ce n’est pas le cas, ce sera une approximation.
    cdelt1 = (alpha_coord[-1] - alpha_coord[0]) / (naxis1 - 1)
    cdelt2 = (beta_coord[-1] - beta_coord[0]) / (naxis2 - 1)
    
    # Référence au centre de la grille
    crpix1 = naxis1 / 2
    crpix2 = naxis2 / 2
    
    crval1 = alpha_coord[naxis1 // 2]
    crval2 = beta_coord[naxis2 // 2]
    
    wcs.wcs.crpix = [crpix1, crpix2]
    wcs.wcs.cdelt = [cdelt1, cdelt2]
    wcs.wcs.crval = [crval1, crval2]
    
    # Type de projection (à adapter selon le cas)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    return wcs

def check_overlap(custom_wcs, shape_source, ref_wcs, shape_ref):
    # Coins pixel dans l'image source
    corners_source = np.array([
        [0, 0],
        [0, shape_source[0]-1],
        [shape_source[1]-1, shape_source[0]-1],
        [shape_source[1]-1, 0]
    ])
    ref_wcs_2d = ref_wcs.celestial
    # Convertir coins source en coordonnées monde (RA, DEC)
    ra, dec = custom_wcs.all_pix2world(corners_source[:,0], corners_source[:,1], 0)
    
    # Convertir RA, DEC en pixel dans ref_wcs
    x_ref, y_ref = ref_wcs_2d.all_world2pix(ra, dec, 0)
    
    print("Coins source en pixels dans WCS cible :")
    for i, (x, y) in enumerate(zip(x_ref, y_ref)):
        print(f"Coin {i}: x={x:.1f}, y={y:.1f}")
    
    # Vérifier si coins sont dans la grille cible
    in_x = np.logical_and(x_ref >= 0, x_ref < shape_ref[1])
    in_y = np.logical_and(y_ref >= 0, y_ref < shape_ref[0])
    inside = np.logical_and(in_x, in_y)
    
    print("Coins dans la grille cible :", inside)
    if not np.any(inside):
        print("Aucun coin source n’est dans la zone cible, il n’y aura pas de recouvrement.")
    else:
        print("Au moins un coin source est dans la zone cible, un recouvrement est possible.")

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_corners(wcs, shape, label=None, color='red', marker='o'):
    """
    Affiche les coins de l'image définie par la WCS et sa shape.

    Parameters:
    -----------
    wcs : astropy.wcs.wcsapi.BaseLowLevelWCS (ex: SlicedFITSWCS)
        WCS 2D de l'image.
    shape : tuple
        Taille de l'image (ny, nx).
    label : str, optional
        Légende du plot.
    color : str, optional
        Couleur des points.
    marker : str, optional
        Type de marqueur matplotlib.
    """
    ny, nx = shape

    # Coordonnées pixel des 4 coins (en pixel coordinates, origine 0)
    corners_pix = np.array([
        [0, 0],          # coin bas-gauche
        [nx - 1, 0],     # coin bas-droit
        [nx - 1, ny - 1],# coin haut-droit
        [0, ny - 1]      # coin haut-gauche
    ])

    # Conversion pixels -> coordonnées célestes (world)
    # Avec l'API WCS 2D, on utilise pixel_to_world_values(*coords)
    # attention à l’ordre x, y (colonnes, lignes)
    ra, dec = wcs.pixel_to_world_values(corners_pix[:,0], corners_pix[:,1])

    plt.plot(ra, dec, marker=marker, linestyle='-', color=color, label=label)
    for i, (x, y) in enumerate(zip(ra, dec)):
        plt.text(x, y, f"C{i}", color=color)

    plt.xlabel("RA (deg)")
    plt.ylabel("DEC (deg)")


def main():
    
    # dir = "/home/nmonnier/Data/JWST/PIPELINE/Point_source/stage2/"
    # fits_files = [
    #     # dir + "jw06659001001_05101_00001_mirifushort_s3d.fits",
    #     # dir + "jw06659001001_05101_00002_mirifushort_s3d.fits",
    #     # dir + "jw06659001001_05101_00003_mirifushort_s3d.fits",
    #     dir + "jw06659001001_05101_00004_mirifushort_s3d.fits",
    # ]
# 
    dir = "/home/nmonnier/Data/JWST/PIPELINE/small_NGC/stage2/"
    fits_files = [
        # dir + "jw01192001001_0310v_00001_mirifushort_s3d.fits",
        # dir + "jw01192001001_0310v_00002_mirifushort_s3d.fits",
        dir + "jw01192001001_0310v_00003_mirifushort_s3d.fits",
        # dir + "jw01192001001_0310v_00004_mirifushort_s3d.fits",
    ]
    fits_file = fits_files[0]
    with fits.open(fits_file) as ref_hdu:
        ref_wcs = WCS(ref_hdu[1].header, ref_hdu)  # Pass HDUList to handle -TAB coordinates
        ref_shape = (ref_hdu[1].header["NAXIS2"], ref_hdu[1].header["NAXIS1"])  # 2D shape

    ndith = [0]  # Dithers to use for the test
    chan_idx = 0  # Channel index to test
    slice_idx = -1  # Last slice index to test
    numpy_slice, alpha_coord, beta_coord, data_dict , spectroModel= parse_options(ndith, chan_idx, slice_idx)
    raise ValueError("STOP")


    # TODEL : Chan 1a
    chan_idx = 1  # Channel index to test
    slice_idx = 0  # Last slice index to test
    numpy_slice_a, _, _, _ , _= parse_options(ndith, chan_idx, slice_idx)

    # alpha_coord = alpha_coord-360 # Ajustement pour correspondre à la projection de l'image FITS
    custom_wcs = create_custom_wcs(alpha_coord, beta_coord)
    shape_source = numpy_slice.shape
    shape_ref = ref_shape
    
    check_overlap(custom_wcs, shape_source, ref_wcs, shape_ref)


    # --- Lecture FITS ---
    slice_idx = -1  # Dernier slice
    fits_slice, fits_wcs_2d = read_fits_slice_and_wcs_2d(fits_file, slice_idx=slice_idx)
    fits_shape = fits_slice.shape

    # --- Créer WCS custom de numpy_slice ---
    custom_wcs = create_custom_wcs_from_coords(alpha_coord, beta_coord)

    # --- Reprojection numpy_slice sur la grille FITS ---
    # Note : fits_wcs_2d est la cible, custom_wcs est la source
    numpy_slice_reprojected, footprint = reproject_interp(
        (numpy_slice, custom_wcs),
        fits_wcs_2d,
        shape_out=fits_shape, 
        return_footprint=True
    )
    print("Footprint min/max:", np.nanmin(footprint), np.nanmax(footprint))
    

    # # Affichage des coins des images
    # plt.figure()
    # plot_corners(fits_wcs_2d, fits_slice.shape, label="FITS slice", color='blue')
    # plot_corners(custom_wcs, numpy_slice.shape, label="Not Weighted")
    # plt.legend()
    # plt.show()


    # Scale both image to the same range
    fits_slice[fits_slice < 0] = 0 
    numpy_slice_reprojected = numpy_slice_reprojected* (np.nanmax(fits_slice)/np.nanmax(numpy_slice_reprojected))


    # # --- Affichage ---
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), subplot_kw={'projection': fits_wcs_2d})
    # # ax.imshow(fits_slice, origin='lower', cmap='gray', alpha=0.7, label="FITS slice")
    # im0 = ax[0, 0].imshow(numpy_slice_reprojected, origin='lower', cmap='plasma', alpha=1, label="Surfh Projected Slice")
    # ax[0, 0].set_title("Surfh Projected Slice")

    # im1 = ax[0, 1].imshow(fits_slice, origin='lower', cmap='plasma', alpha=1, label="FITS Slice")
    # ax[0, 1].set_title("FITS Slice")
    
    # im2 = ax[1, 0].imshow(numpy_slice_reprojected - fits_slice, origin='lower', cmap='plasma', alpha=1, label="Surfh Original Slice")
    # ax[1, 0].set_title("Difference (Surfh - FITS)")

    # # relative difference
    # relative_difference = 100 * (numpy_slice_reprojected - fits_slice) / fits_slice
    # relative_difference[np.isnan(relative_difference)] = 0  # Remplacer NaN par 0
    # relative_difference[relative_difference > 200] = 200
    # im3 = ax[1, 1].imshow(relative_difference, origin='lower', cmap='plasma', alpha=1, label="Relative Difference")
    # ax[1, 1].set_title("Relative Difference (%)")

    # # Centre FITS (optionnel, pour véri fication)
    # crpix1 = fits_slice.shape[1] / 2
    # crpix2 = fits_slice.shape[0] / 2
    # ax[0, 0].plot(crpix1, crpix2, marker='x', color='white', markersize=10, label='FITS center')

    # # for dith in range(4):
    # #     alpha, beta = data_dict['target']['2a'][dith]
    # #     print(f"Dither {dith+1} - Alpha: {alpha}, Beta: {beta}")
    # #     ax[0, 0].plot(alpha, beta, marker='+', color='red', markersize=12, label='Target (alpha, beta)')


    # # ax.legend()
    
    
    # plt.colorbar(im0,ax=ax[0, 0])
    # plt.colorbar(im1,ax=ax[0, 1])
    # plt.colorbar(im2,ax=ax[1, 0])
    # plt.colorbar(im3,ax=ax[1, 1])

    # plt.show()

    extent = [alpha_coord.min(), alpha_coord.max(), beta_coord.min(), beta_coord.max()]

    plt.figure()
    plt.imshow(numpy_slice, origin='lower', cmap='viridis', extent=extent, alpha=0.5)
    plt.imshow(numpy_slice_a, origin='lower', cmap='viridis', extent=extent, alpha=0.5)
    plt.colorbar()
    
    plt.figure()
    plt.imshow(numpy_slice, origin='lower', cmap='viridis', extent=extent)
    plt.figure()
    plt.imshow(numpy_slice_a, origin='lower', cmap='viridis', extent=extent)

    fig, ax = plt.subplots(nrows=2)
    

    numpy_slice[np.where(numpy_slice==np.nanmax(numpy_slice))] = np.nanmax(numpy_slice)  # Set max value to NaN for better visualization
    numpy_slice_a[np.where(numpy_slice_a==np.nanmax(numpy_slice_a))] = np.nanmax(numpy_slice_a)  # Set max value to NaN for better visualization
    ax[0].imshow(numpy_slice, origin='lower', cmap='viridis', extent=extent, alpha=1)
    ax[1].imshow(numpy_slice_a, origin='lower', cmap='viridis', extent=extent, alpha=1)
    # plt.colorbar()
    for dith in range(4):
        alpha, beta = spectroModel.pointings[1][dith].alpha, spectroModel.pointings[1][dith].beta
        print(f"Dither {dith+1} - Alpha: {alpha}, Beta: {beta}")
        plt.plot(alpha, beta, marker='+', color='red', markersize=12, label='Target (alpha, beta)')
    plt.show()

    tmp = np.zeros((numpy_slice.shape[0], numpy_slice.shape[1]))
    cube = np.stack((numpy_slice, tmp, tmp, tmp, tmp, numpy_slice_a, tmp, tmp), axis=0)
    print("Cube shape = ", cube.shape)
    # cube_vizualisation.plot_cube(cube, np.arange(cube.shape[0]))

if __name__ == "__main__":
    main()
