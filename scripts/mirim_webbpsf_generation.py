import numpy as np
import os
import matplotlib.pyplot as plt
import webbpsf
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs
from astropy.io import fits



def compute_monochromatic_psfs2(wave_filter, oversample=4, pixelscale=0.11, fov_arcsec=10, norm='last', date=None):
    """
    https://pythonhosted.org/webbpsf/webbpsf.html#psf-normalization
    This function use webbpsf tool to simulate monochromatic psf
    how to use :
    psfOverSamp = 1 # detector resolution
    psfFov = 5      # arcsec
    psf_size = int(round(psfFov/MIRI.pixelscale)*psfOverSamp)
    psfPath = "PSFs/detectorRes/"
    #psfs_monochromatic = compute_monochromatic_psfs(lamAllMIRI, psfFov, psf_size)
    np.save('PSFs/detectorRes/psfMonochromatic_AlainAbergelWave_crop.npy', psf_obj)
    psf = phd.compute_monochromatic_psfs(wave, psf_param['fov'], psf_param['size'])
    """
    miri = webbpsf.MIRI()
    miri.mode = 'IFU'
    miri.band= '1C'
    miri.pixelscale = pixelscale
    if date is not None:
        miri.load_wss_opd_by_date(date, plot=False, choice="closest")


    psf_number = len(wave_filter)
    psfs_monoch = []
        
    miri._rotation = 0.0 # rotation de la psf dans le ref du télescope, plan V2/V3
    
    for i in range(psf_number):
        #        print(wave_filter[i]*1e-6)
        miri.options["output_mode"] = "detector sampled"
        psf_file = miri.calc_psf(monochromatic=wave_filter[i] * 1e-6,
                            oversample=oversample,
                            normalize=norm,
                            fov_arcsec=fov_arcsec)
        # print(psf_file[0].data.shape)
        psfs_monoch.append(psf_file[0].data)
        
        # print(i)
        if (i+1)%10 == 0:
            print("{} / {}".format(i, len(wave_filter)))
            
    return psfs_monoch



# array of wavelength 
# wavel_axis = np.load('/home/nmonnier/Data/JWST/Orion_bar/Observation_1/Fusion/Templates/wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy') #wavelength_mrs.get_mrs_wavelength('1c')
# spectrums = fits.open("/home/nmonnier/Data/JWST/Simulation/Orion/Templates/spectra_mir_orion.fits")[1].data
# wavel_axis = spectrums.wavelength
# wavel_axis = wavel_axis[::4] # SS 
wavel_axis = np.load('/home/nmonnier/Data/JWST/NGC_7023/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4ABC.npy')

oversample = 1
# Pixel scale
pixelscale = 0.1 # valeur choisie pour le cas de test
# Size of PSF in pixel (here 501x501)
nb_pixels=125
fov_arcsec = pixelscale * nb_pixels

norm = 'last'
# Time code of observation
date = "2023-09-25T22:02:45"
psf_number = len(wavel_axis)

miri = webbpsf.MIRI()
miri.mode = 'imaging'
miri.pixelscale = pixelscale
PSF = list()
for i in range(psf_number):
    psf_file = miri.calc_psf(monochromatic=wavel_axis[i] * 1e-6,
                             oversample=oversample,
                            normalize=norm,
                            fov_arcsec=fov_arcsec)
    PSF.append(psf_file[0].data)
    # print(i)
    if (i+1)%10 == 0:
        print("{} / {}".format(i, len(wavel_axis)))

# psfs_monoch = compute_monochromatic_psfs2(wavel_axis, oversample=oversample, pixelscale=pixelscale, fov_arcsec=fov_arcsec, norm=norm, date=date)

psfs_monoch_array = np.array(PSF)


file_path = "/home/nmonnier/Data/JWST/NGC_7023/Fusion/PSF/"
file_name = f"mirim_psfs_pixscale{pixelscale}_npix_{nb_pixels}.npy"
np.save(file_path + file_name, psfs_monoch_array)