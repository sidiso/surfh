import numpy as np
from astropy.io import fits



def save_numpy_to_fits(data, metadata, wavelengths, filename):
    """Save a numpy array to a FITS file with the given header."""
    hdu = fits.PrimaryHDU(data=data)
    header = hdu.header

    # Build header metadata
    header['AUTHOR'] = 'Nicolas Monnier'
    header['NWAVEL'] = data.shape[0]   # Number of wavelength points
    header['NAXIS1'] = data.shape[2]   # Number of alpha
    header['NAXIS2'] = data.shape[1]   # Number of beta

    header['PA_V3'] = metadata['PA_V3']   # Position Angle (V3) in degrees
    header['TARG_RA'] = metadata ['TARG_RA']  # Target Right Ascension (
    header['TARG_DEC'] = metadata['TARG_DEC']   # Target Declination (in degrees)
    header['RA_V1'] = metadata['RA_V1']   # Target RA in V1 frame (in degrees)
    header['DEC_V1'] = metadata['DEC_V1']   # Target DEC in V1
    header['RA_REF'] = metadata['RA_REF']   # Reference RA (in degrees)
    header['DEC_REF'] = metadata['DEC_REF']   # Reference DEC (in degrees)

    # --- Axe spectral ---
    if np.allclose(np.diff(wavelengths), np.diff(wavelengths)[0], rtol=1e-6):
        # Axe régulier
        print("Regular wavelength axis")
        dw = wavelengths[1] - wavelengths[0]
        header['CTYPE3'] = 'WAVE'
        header['CUNIT3'] = 'um'
        header['CRVAL3'] = wavelengths[0]
        header['CDELT3'] = dw
        header['CRPIX3'] = 1
    else:
        print("Irregular wavelength axis")
        # --- Axe spectral via WCS -TAB ---
        # FITS: pour un cube (nw, ny, nx), l'axe spectral correspond à l'axe FITS 3
        header['CTYPE3'] = 'WAVE-TAB'   # type d'axe + algorithme TAB
        header['CUNIT3'] = 'um'         # doit matcher TUNITn de la table
        header['PS3_0']  = 'WCS-TAB'    # EXTNAME de la table
        header['PS3_1']  = 'WAVELENGTH' # nom de colonne qui contient le vecteur lambda
        header['PV3_1']  = 1            # EXTVER de la table (par défaut 1)
        # Ces 3 là ne sont pas requis par -TAB, mais certains softs aiment les avoir:
        header['CRPIX3'] = 1
        header['CRVAL3'] = 1
        header['CDELT3'] = 1

        # 2) Table WCS-TAB : une ligne, une colonne "WAVELENGTH" contenant tout le vecteur
        wl = np.asarray(wavelengths, dtype=np.float64)
        # Astropy attend un tableau 2D (nrows, repeat) pour faire une "cellule-vecteur"
        wl_cell = wl.reshape(1, -1)
        col = fits.Column(name='WAVELENGTH',
                        format=f'{wl.size}D',  # vecteur de longueur N (type float64)
                        unit='um',
                        array=wl_cell)
        wave_hdu = fits.BinTableHDU.from_columns([col], name='WCS-TABLE')
        wave_hdu.header['EXTVER'] = 1

        # 3) Écriture
        fits.HDUList([hdu, wave_hdu]).writeto(filename, overwrite=True)
        return
    hdu.writeto(filename, overwrite=True)

def load_fits_metadata(fits_path):
    """Load metadata from a FITS file header."""
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        metadata = {
            'NWAVEL': hdr.get('NWAVEL', None),
            'NAXIS1': hdr.get('NAXIS1', None),
            'NAXIS2': hdr.get('NAXIS2', None),
            'PA_V3': hdr.get('PA_V3', None),
            'TARG_RA': hdr.get('TARG_RA', None),
            'TARG_DEC': hdr.get('TARG_DEC', None),
            'RA_V1': hdr.get('RA_V1', None),
            'DEC_V1': hdr.get('DEC_V1', None),
            'RA_REF': hdr.get('RA_REF', None),
            'DEC_REF': hdr.get('DEC_REF', None),
        }
    return metadata




path_file = '/home/nmonnier/Data/JWST/NGC_7023/Fusion/'
res_dir = 'Results/lcg_MC_7_MO_4_Temp_6_nit_500_mu_5.00e+06_SD_True/'

ref_file = path_file+ "/Filtered_slices/ch1a_00001_corrected_filtered.fits"
metadata = load_fits_metadata(ref_file)

data = np.load(path_file + res_dir + 'res_cube.npy')
wavelengths = np.load(path_file + res_dir + 'wavel.npy')
print(wavelengths)
print(wavelengths.shape)
print(data.shape)

save_numpy_to_fits(data, metadata, wavelengths, path_file + res_dir + 'res_cube.fits')