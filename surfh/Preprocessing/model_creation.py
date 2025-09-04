
import numpy as np
import os
import udft

from rich import print

from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits

from surfh.Models import wavelength_mrs, instru, metadataMRS
from surfh.Models import spectroModel
import matplotlib.pyplot as plt

def get_axis(model):
    return model.alpha_axis, model.beta_axis

def create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape):
    """Create the spectrograph model."""
    main_pointing = instru.Coord(0, 0)
    pointings = []
    # Delta offsets for each channel based on the pointing reference ch1
    delta_pointing = {'1a':(0,0),
                      '1a':(0.087/3600, 0.288/3600),
                      '1a':(0.654/3600, 0.002/3600),
                      '1a':(-0.644/3600, -0.261/3600)}

    for idx, chan in enumerate(instruments.keys()):
        # DETLA_RA, DELTA_DEC = metadataMRS.get_chan_delta_pointing(chan)
        # RA_CORR, DEC_CORR = metadataMRS.get_pointing_correction_SN2023fyq(chan)
        # RA - DITH_RA seems so be the working solution here
        # pointing_chan = [main_pointing + instru.Coord(-RA_CORR - DITH_RA + DETLA_RA, -DEC_CORR + DITH_DEC + DELTA_DEC) for (RA, DEC), (DITH_RA, DITH_DEC) in zip(data_dict['target'][chan], data_dict['dither'][chan])]
        RA_CORR, DEC_CORR = metadataMRS.get_band_delta_pointing(chan)
        pointing_chan = [main_pointing + instru.Coord( -RA_CORR - DITH_RA, -DEC_CORR + DITH_DEC) for (RA, DEC), (DITH_RA, DITH_DEC) in zip(data_dict['target'][chan], data_dict['dither'][chan])]
        pointings.append(instru.CoordList(pointing_chan).pix(step_angle))


    print("!!!!!!!!!!!!!!!!!!!")
    print(pointings)
    print("!!!!!!!!!!!!!!!!!!!")
    origin_alpha_axis = (np.arange(imshape[0]) * step_angle - np.mean(np.arange(imshape[0]) * step_angle))
    origin_beta_axis = np.arange(imshape[1]) * step_angle - np.mean(np.arange(imshape[1]) * step_angle)

    # TODO : Warning here Mean is 0 because the pointings are centered on 0,0
    mean_alpha = np.mean([pointings[-1][dith].alpha for dith in range(4)])
    mean_beta = np.mean([pointings[-1][dith].beta for dith in range(4)])
    # mean_alpha = 0#np.mean([data_dict['target']['1a'][dith][0] for dith in range(4)])
    # mean_beta = 0#np.mean([data_dict['target']['1a'][dith][1] for dith in range(4)])

    # mean_alpha = data_dict['target']['1a'][0][0] 
    # mean_beta = data_dict['target']['1a'][0][1] 
    alpha_axis = origin_alpha_axis + mean_alpha
    beta_axis = origin_beta_axis + mean_beta
    print(alpha_axis, beta_axis)

    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)

def create_instruments(data_dict, list_chan):
    """Create instrument configurations for each channel."""
    instruments = {}

    # Define the channel specifications
    # Each tuple contains (n_slit, r_min, r_max, det_pix_size, fov_x, fov_y)
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
    data_dict = {'data': {}, 'target': {}, 'targetV1' :{}, 'targetREF': {}, 'dither': {}, 'rotation': {}, 'PA_V3': {}}

    for chan in list_chan:
        data_dict['data'][chan] = []
        data_dict['target'][chan] = []
        data_dict['targetV1'][chan] = []
        data_dict['targetREF'][chan] = []
        data_dict['dither'][chan] = []
        data_dict['rotation'][chan] = 0.

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

                    data_dict['data'][chan].append(ndata)
                    data_dict['target'][chan].append((TARG_RA, TARG_DEC))
                    data_dict['targetV1'][chan]= (RA_V1, DEC_V1)
                    data_dict['targetREF'][chan] = (RA_REF, DEC_REF)
                    data_dict['dither'][chan].append((DITHER_RA, DITHER_DEC))
                    data_dict['rotation'][chan] = metadataMRS.get_MRS_rotation(chan)
                    data_dict['PA_V3'][chan] = PA_V3
    return data_dict

# TODO: Change this function, almost useless now. alpha and beta axis are now created in the model creation
def load_simulation_data(paths, list_chan):
    """Load simulation data."""
    ref_wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_SS4.npy')) # Reference wavelength for chan 1234-ABC with SS4
    # templates = np.load(os.path.join(paths['template_dir'], 'nmf_SN2023fyq_1ABC_2ABC_3ABC_4ABC_6_templates_SS4.npy'))
    # wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_SN2023fyq_1ABC_2ABC_3ABC_4AB_SS4.npy'))
    # templates = np.load(os.path.join(paths['template_dir'], 'nmf_SN2023fyq_1ABC_2ABC_3ABC_4AB_6_templates_SS4.npy'))
    # wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_NGC7023_1C_2ABC_3ABC_SS4.npy'))
    # templates = np.load(os.path.join(paths['template_dir'], 'nmf_NGC7023_1C_2ABC_3ABC_6_templates_SS4.npy'))
    # wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_NGC7023_1C_2ABC_3ABC_4AB_SS4.npy'))
    # templates = np.load(os.path.join(paths['template_dir'], 'nmf_NGC7023_1C_2ABC_3ABC_4AB_6_templates_SS4.npy'))
    wavel_axis = np.load(os.path.join(paths['template_dir'], 'wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB_SS4.npy'))
    templates = np.load(os.path.join(paths['template_dir'], 'nmf_NGC7023_1ABC_2ABC_3ABC_4AB_6_templates_SS4.npy'))


    otf = np.load(os.path.join(paths['psf_dir'], 'psfs_pixscale0.1_npix_125_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy'))
    imshape = (otf.shape[1], otf.shape[2])

    # Sort wavelegnth regarding the channel list
    indexes = np.where((ref_wavel_axis>wavelength_mrs.get_mrs_wavelength(list_chan[0])[0]) & (ref_wavel_axis<wavelength_mrs.get_mrs_wavelength(list_chan[-1])[-1]))[0]
    if indexes[0] == 0:
        window_slice = slice(indexes[0], indexes[-1] +1, None) # If the first index is 0, take it
    else:
        window_slice = slice(indexes[0]-1, indexes[-1] +1, None) # 

    print(f"Window slice is {window_slice}")

    # wavel_axis = wavel_axis[window_slice]
    otf = otf[window_slice]

    # otf = otf[:-25,:,:]  # Remove the last 25 slices, they are not used in the simulation
    sotf = udft.ir2fr(otf, imshape)

    print(f"Wavel axis is {wavel_axis}")
    print(wavel_axis.shape)
    print(sotf.shape)

    return wavel_axis, templates, sotf


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
