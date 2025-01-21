
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

import os
#from surfh import shared_dict

#from surfh.AsyncProcessPoolLight import APPL

from astropy.io import fits
# %% 1) INPUTS FOR MODELS

import numpy as np
import matplotlib.pyplot as plt
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.calc_utils import build_default_calc
from matplotlib import cm

from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.calc_utils import build_default_calc
from matplotlib import cm
from pathlib import Path


# update matplotlib font
plt.rcParams.update({'font.size': 6})



fits_directory = '/home/nmonnier/Data/JWST/Orion_bar/Single_fits'
pce_directory = '/home/nmonnier/Data/JWST/Orion_bar/All_bands_pce'

telescope = 'jwst'
instrument = 'miri'
mode = 'mrs'
calculation = build_default_calc(telescope, instrument, mode)

# Photon conversion efficiency plotter
channels = ['ch1', 'ch2', 'ch3', 'ch4']
bands = ['short', 'medium', 'long']

config = {
  'instrument': {'aperture': 'ch1',
  'disperser': 'short',
  'filter': None,
  'instrument': 'miri',
  'mode': 'mrs'}}

colors = iter(cm.rainbow(np.linspace(0, 1, len(channels) * len(bands))))

efficiencies = []

fig, ax = plt.subplots(1, 1, figsize = (10, 2), dpi = 500)



for file in os.listdir(fits_directory):

    name = file.split('_')
    chan = name[1] + name[2]
    band = name [3]

    hdul = fits.open(fits_directory + '/' + file)
    hdr = hdul[1].header
    wavel = (np.arange(hdr['NAXIS3']) +hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']

    color = next(colors)
    config['instrument']['aperture'] = chan
    config['instrument']['disperser'] = band
    instrument_factory = InstrumentFactory(config = config)
    efficiency = instrument_factory.get_total_eff(wavel)

    np.save(pce_directory + '/' + Path(file).stem + '.pce.npy', efficiency)

    efficiencies.append(efficiency)
    ax.plot(wavel, efficiency, color = color, linestyle = '-', linewidth = 0.2, alpha = 0.7)
    ax.plot([], [], color = color, linestyle = '-', linewidth = 1.5, alpha = 0.7, label = f'{chan}{band}')
    ax.fill_between(wavel, efficiency, color = color, alpha = 0.7)



plt.show()


















    