


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
