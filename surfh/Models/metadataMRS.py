
import pysiaf
import os
import numpy as np


def get_dithering_example(chan=None, dithering=None):
    """Return specific dithering pointing based on hard-coded simulation offsets."""
    if chan is None or dithering is None:
        raise RuntimeError("Error: dithering or chan is None!")

    # Toutes les données hardcodées
    data = {
        "1a": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "1b": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "1c": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },

        "2a": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "2b": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "2c": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },

        "3a": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "3b": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "3c": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },

        "4a": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "4b": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
        "4c": {
            1: (-2.0780633027612e-05, 0.000343680211210981),
            2: (-0.00013700243665798, 0.000600239378139109),
            3: (-9.9744615649607e-05, 0.000331837711337799),
            4: (-5.8111788016598e-05, 0.000611752155714826),
        },
    }

    chan = chan.lower()

    if chan not in data:
        raise RuntimeError(f"Unknown channel: {chan}")

    if dithering not in data[chan]:
        raise RuntimeError(f"Dithering {dithering} not available for channel {chan}")

    return data[chan][dithering]


def build_MRS_resolving_power(chan=None, resolving_path=None, wavelength=None):
    """ Build Resolving power for specific band (and wavelength array) from sub-sampled resolving power. """
    if resolving_path is None or chan is None:
        raise RuntimeError("Error resolving path or chan is set to None !")
    for file in os.listdir(resolving_path):
        if chan.upper() in file:
            data = np.loadtxt(resolving_path + file, delimiter=',')
            SS_wavel = data[:,0]
            SS_power = data[:,1]
            resolving_power = np.interp(wavelength, SS_wavel, SS_power)
            print("Done")
    return resolving_power


def get_MRS_rotation(chan=None):
    """Get the rotation angle for MRS channels."""
    if chan is None:
        raise ValueError("Channel must be specified.")
    
    if chan in ['1a', '1b', '1c']:
        return 8.4
    elif chan in ['2a', '2b', '2c']:
        return 8.2
    elif chan in ['3a', '3b', '3c']:
        return 7.5
    elif chan in ['4a', '4b', '4c']:
        return 8.3
    else:
        raise ValueError(f"Unknown channel: {chan}")
    

def get_band_delta_pointing(chan=None):
    """Get the (RA, DEC) delta coordinate from the based on reference point : center of chan 1a"""
    if chan is None:
        raise ValueError("Channel must be specified")

    instrument = 'MIRI'
    siaf = pysiaf.Siaf(instrument)
    mrs_aper_ch = [x for x in siaf.apertures.keys() if 'MIRIFU_CHANNEL' in x]

    v3_ref = siaf['MIRIFU_CHANNEL1A'].V3Ref
    v2_ref = siaf['MIRIFU_CHANNEL1A'].V2Ref

    search_term =  chan
    matches = [item for item in mrs_aper_ch if search_term.lower() in item.lower()]

    aperture = siaf[matches[0]]

    v3_center = aperture.V3Ref
    v2_center = aperture.V2Ref

    return  ((v2_center - v2_ref) / 3600, (v3_center - v3_ref) / 3600)


def get_chan_delta_pointing(chan=None):
    """Get the (RA, DEC) delta coordinate from the based on reference point : center of chan 1"""
    if chan is None:
        raise ValueError("Channel must be specified")
    
    if chan in ['1a', '1b', '1c']:
        return (0,0)
    elif chan in ['2a', '2b', '2c']:
        return (0.087/3600, 0.288/3600)
    elif chan in ['3a', '3b', '3c']:
        return (0.654/3600, 0.002/3600)
    elif chan in ['4a', '4b', '4c']:
        return (-0.644/3600, -0.261/3600)
    else:
        raise ValueError(f"Unknown channel {chan}")


def get_pointing_correction_SN2023fyq(chan=None):
    # Pointing correction for SN2023fyq
    pointing_correction = {'1a' : (0, 0),  '1b' : (-1.00E-07, -2.70E-05),  '1c' : (2.72E-05, 5.48E-05),  # ch1
                           '2a' : (-2.81E-05, 2.84E-05), '2b' : (-2.78E-05, -5.28E-05), '2c' : (3.47E-05, -4.86E-05),  # ch2
                           '3a' : (-1.20E-06, 1.20E-06), '3b' : (-1.90E-06, -6.73E-05), '3c' : (-1.00E-06,-5.60E-05),  # ch3
                           '4a' : (-1.07E-04, -2.91E-04), '4b' : (-1.10E-04,-3.23E-04), '4c' : (-1.40E-04,-2.86E-04)} # ch4
    
    return pointing_correction[chan]


def get_FoV_offset(chan=None):
    if chan is None:
        raise ValueError("Channel must be specified")
    
    if chan == '1a':
        return (0,0)