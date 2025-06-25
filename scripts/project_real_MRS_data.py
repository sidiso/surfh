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
    idx=0

    coords = []
    for file in sorted(os.listdir(save_filter_corrected_dir)):
        for chan in list_chan:
            if 'ch1a' in file:
                with fits.open(os.path.join(save_filter_corrected_dir, file)) as hdul:
                    header = hdul[0].header
                    TARG_RA = header['TARG_RA']     
                    TARG_DEC = header['TARG_DEC']       
                    coords.append((TARG_RA, TARG_DEC))      

    permutations = list(itertools.permutations(coords))
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

    idx = 1# chan/band number (0 for 1a, 1 for 1b, 2 for 1c, etc.)
    # weigthed_proj, not_weigthed = spectroModel.plot_slice(ndata, idx, 50)
    # not_weigthed = spectroModel.test_project_mrs_slice(ndata, idx, 900)
    # print(f"not_weigthed.shape = {not_weigthed.shape}")

    # extent = [
    #     alpha_coord[0], alpha_coord[-1],
    #     beta_coord[0], beta_coord[-1]   
    # ]
    # print(f"extent = {extent}")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(not_weigthed, aspect='auto')



    i=0
    idx=0

    coords = []
    print("sorted files = ",sorted(os.listdir(paths["save_filter_corrected_dir"])))
    for file in sorted(os.listdir(paths["save_filter_corrected_dir"])):

        if 'ch1a' in file:
            print("File is ", file)
            with fits.open(os.path.join(paths["save_filter_corrected_dir"], file)) as hdul:
                header = hdul[0].header
                TARG_RA = header['TARG_RA']     
                TARG_DEC = header['TARG_DEC']       
                coords.append((TARG_RA, TARG_DEC))      

    permutations = list(itertools.permutations(coords))
    print("permutations = ", len(permutations))
    print(coords)
    not_weigthed = spectroModel.test_project_mrs_slice(ndata, 1, -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.flipud(not_weigthed), aspect='auto')
    plt.title(f"Original projection for ch2a")
    plt.show()

    main_pointing = instru.Coord(0, 0)
    for idx in range(len(permutations)):
        pointings = []
        for i in range(2):
            pointing_chan = [main_pointing + instru.Coord(permutations[idx][dith][0], permutations[idx][dith][1]) for dith in range(4)]
            pointings.append(instru.CoordList(pointing_chan).pix(step_angle))
        print(f"!?!?!? : pointings = {pointings}")
        spectroModel.pointings = instru.CoordList(pointings).pix(step_angle)
        not_weigthed = spectroModel.test_project_mrs_slice(ndata, 1, -1)
        plt.figure(figsize=(10, 10))
        # plt.imshow(np.fliplr(np.flipud(not_weigthed)), aspect='auto')
        plt.imshow(np.fliplr(not_weigthed), aspect='auto')
        plt.title(f"Weighted projection for ch2a - permutations {idx}")
        plt.show()

    # plt.colorbar()
    # plt.title(f"Weighted projection for {list_chan[idx]}")
    # # Midpoints in physical coordinates
    # mid_alpha = (alpha_coord[0] + alpha_coord[-1]) / 2
    # mid_beta = (beta_coord[0] + beta_coord[-1]) / 2
    # plt.axvline(x=mid_alpha, color='red', linestyle='--', label='Vertical Midline')
    # plt.axhline(y=mid_beta, color='blue', linestyle='--', label='Horizontal Midline')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    parse_options()