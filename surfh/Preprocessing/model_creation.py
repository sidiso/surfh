
import numpy as np
import os
import udft

from rich import print

from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits

from surfh.Models import wavelength_mrs, instru, metadataMRS
from surfh.Models import spectroModel, MiriModel
from surfh.Others.context import Config
import matplotlib.pyplot as plt

def get_axis(model):
    return model.alpha_axis, model.beta_axis

def tmp_create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape, xshift=0, yshift=0, ch='4a'):
    """Create the spectrograph model."""
    main_pointing = instru.Coord(0, 0)
    pointings = []

    for idx, chan in enumerate(instruments.keys()):
        if chan == ch:
            x_shift_chan = xshift*step_angle
            y_shift_chan = yshift*step_angle
        else:
            x_shift_chan = 0
            y_shift_chan = 0

        RA_CORR, DEC_CORR = metadataMRS.get_band_delta_pointing(chan)
        pointing_chan = [main_pointing + instru.Coord( -RA_CORR - DITH_RA - x_shift_chan, -DEC_CORR + DITH_DEC + y_shift_chan) for (RA, DEC), (DITH_RA, DITH_DEC) in zip(data_dict['target'][chan], data_dict['dither'][chan])]
        pointings.append(instru.CoordList(pointing_chan).pix(step_angle))


    origin_alpha_axis = (np.arange(imshape[0]) * step_angle - np.mean(np.arange(imshape[0]) * step_angle))
    origin_beta_axis = np.arange(imshape[1]) * step_angle - np.mean(np.arange(imshape[1]) * step_angle)

    # TODO : Warning here Mean is 0 because the pointings are centered on 0,0
    mean_alpha = np.mean([pointings[-1][dith].alpha for dith in range(4)])
    mean_beta = np.mean([pointings[-1][dith].beta for dith in range(4)])
    alpha_axis = origin_alpha_axis + mean_alpha
    beta_axis = origin_beta_axis + mean_beta

    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)


def create_model(sotf, templates, wavel_axis, instruments, step_angle, data_dict, imshape):
    """Create the spectrograph model."""
    main_pointing = instru.Coord(0, 0)
    pointings = []

    for idx, chan in enumerate(instruments.keys()):
        RA_CORR, DEC_CORR = metadataMRS.get_band_delta_pointing(chan)
        pointing_chan = [main_pointing + instru.Coord( -RA_CORR - DITH_RA, -DEC_CORR + DITH_DEC) for (RA, DEC), (DITH_RA, DITH_DEC) in zip(data_dict['target'][chan], data_dict['dither'][chan])]
        pointings.append(instru.CoordList(pointing_chan).pix(step_angle))


    origin_alpha_axis = (np.arange(imshape[0]) * step_angle - np.mean(np.arange(imshape[0]) * step_angle))
    origin_beta_axis = np.arange(imshape[1]) * step_angle - np.mean(np.arange(imshape[1]) * step_angle)

    # TODO : Warning here Mean is 0 because the pointings are centered on 0,0
    mean_alpha = np.mean([pointings[-1][dith].alpha for dith in range(4)])
    mean_beta = np.mean([pointings[-1][dith].beta for dith in range(4)])
    alpha_axis = origin_alpha_axis + mean_alpha
    beta_axis = origin_beta_axis + mean_beta

    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)

def create_instruments(data_dict, config: Config):
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


    signe_rotation = -1 if config.MRS.inverse_rotation else 1
    for chan, (n_slit, r_min, r_max, det_pix_size, fov_x, fov_y) in channel_specs.items():
        if chan in config.MRS.list_channels:
            spec_blur = instru.SpectralBlur(np.mean([r_min, r_max]), None)
            instruments[chan] = instru.IFU(
                fov=instru.FOV(fov_x, fov_y, origin=instru.Coord(0, 0), angle= signe_rotation*data_dict['rotation'][chan]),
                det_pix_size=det_pix_size,
                n_slit=n_slit,
                w_blur=spec_blur,
                pce=None,
                wavel_axis=wavelength_mrs.get_mrs_wavelength(chan),
                name=chan.upper()
            )
    return instruments

def load_data_mirim(list_filter, mirim_data_path):
    """Load MIRI MRS data for the specified filters."""
    data_dict = {'data': {}}
    for file in sorted(os.listdir(mirim_data_path)):
        for filter in list_filter:
            if filter in file:
                with fits.open(os.path.join(mirim_data_path, file)) as hdul:
                    print(f"Loading data for filter {filter} from file {file}")
                    data = hdul[0].data
                    data_dict['data'][filter] = [data]
    return data_dict

def load_mrs_data(config: Config):
    """Load data for the specified channels."""
    data_dict = {'data': {}, 'target': {}, 'targetV1' :{}, 'targetREF': {}, 'dither': {}, 'rotation': {}, 'PA_V3': {}}

    for chan in config.MRS.list_channels:
        data_dict['data'][chan] = []
        data_dict['target'][chan] = []
        data_dict['targetV1'][chan] = []
        data_dict['targetREF'][chan] = []
        data_dict['dither'][chan] = []
        data_dict['rotation'][chan] = 0.

    print("Order of channels loading : ")
    for file in sorted(os.listdir(config.configuration.data_dir)):
        for chan in config.MRS.list_channels:
            if chan in file:
                with fits.open(os.path.join(config.configuration.data_dir, file)) as hdul:
                    print(f"Loading !! data for channel {chan} from file {config.configuration.data_dir}/{file}")
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
def load_mrs_simulation_data(config: Config):
    ref_wavel_axis = np.load(config.configuration.reference_wavelength_file) 
    wavel_axis = np.load(config.configuration.wavelength_file)
    templates = np.load(config.configuration.templates_file)
    otf = np.load(config.configuration.mrs_psf_file)

    imshape = (otf.shape[1], otf.shape[2])

    # Sort wavelegnth regarding the channel list
    indexes = np.where((ref_wavel_axis>wavelength_mrs.get_mrs_wavelength(config.MRS.list_channels[0])[0]) & (ref_wavel_axis<wavelength_mrs.get_mrs_wavelength(config.MRS.list_channels[-1])[-1]))[0]
    if indexes[0] == 0:
        window_slice = slice(indexes[0], indexes[-1] +1, None) # If the first index is 0, take it
    else:
        window_slice = slice(indexes[0]-1, indexes[-1] +2, None) # 

    otf = otf[window_slice]
    sotf = udft.ir2fr(otf, imshape)
    return wavel_axis, templates, sotf

def initialize_fusion_parameters(config):
    step_angle = Angle(config.cube.pixel_resolution, u.arcsec).degree
    return step_angle


def create_miri_model(psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, pixel_arcsec, precompute_H_freq):
    return MiriModel.Mirim_Model_LMM(psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, pixel_arcsec, precompute_H_freq)

def load_miri_simulation_data(paths, list_filter, wavelength):
    ref_list_filter = ['F0560W', 'F0770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
    otf = np.load(os.path.join(paths['psf_dir'], 'mirim_psfs_pixscale0.1_npix_125.npy'))
    imshape = (otf.shape[1], otf.shape[2])

    # Select PSF regarding list_filter
    indexes = [i for i, val in enumerate(ref_list_filter) if val in list_filter]
    otf = otf[:len(wavelength)]
    sotf = udft.ir2fr(otf, imshape)

    # Load PCE -- Don't deal with other multiple wavel now
    list_pce = []
    for file in sorted(os.listdir(os.path.join(paths['miri_filter']))) :
        print(f'Load PCE file for from file {file} ')
        list_pce.append(np.load(os.path.join(paths['miri_filter'])+file)[0])
    pce = np.array(list_pce)
    pce = pce[indexes, :len(wavelength)]
    # Try to load H_freq if exists 
    try:
        H_freq = np.load(os.path.join(paths['template_dir'], 'H_freq.npy'))
    except:
        H_freq = None

    return sotf, pce, H_freq