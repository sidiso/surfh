import numpy as np
import os
import matplotlib.pyplot as plt
import webbpsf
from surfh.Simulation import simulation_data
from surfh.Models import wavelength_mrs



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


sim_dir_path='/home/nmonnier/Projects/JWST/MRS/surfh/cube_orion/'
# _, _, wavel_axis, _, _, _ = simulation_data.get_simulation_data(4, 0, sim_dir_path)

wavel_axis = np.load('/home/nmonnier/Data/JWST/Orion_bar/Fusion/Templates/wavel_axis_orion_1ABC_2ABC_3ABC_4ABC_4_templates_SS4.npy') #wavelength_mrs.get_mrs_wavelength('1c')

oversample = 1
pixelscale = 0.025 # valeur choisie pour le cas de test
nb_pixels=501
fov_arcsec = pixelscale * nb_pixels

norm = 'last'
date = "2023-01-30T01:16:11"

psfs_monoch = compute_monochromatic_psfs2(wavel_axis, oversample=oversample, pixelscale=pixelscale, fov_arcsec=fov_arcsec, norm=norm, date=date)

psfs_monoch_array = np.array(psfs_monoch)

file_path = "/home/nmonnier/Data/JWST/Orion_bar/Fusion/PSF/"
file_name = f"psfs_pixscale{pixelscale}_npix_{nb_pixels}_fov{fov_arcsec}_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy"
np.save(file_path + file_name, psfs_monoch_array)