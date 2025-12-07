from astropy.io import fits
from scipy.ndimage import rotate
import numpy as np


def corrected_slices_to_fits(corrected_slices, rotation, target_RA, target_DEC, filename, selected_chan, slices_shape):

    PA_V3 = rotation
    TARG_RA = target_RA
    TARG_DEC = target_DEC

    # Create a PrimaryHDU object to store the 2D image data
    hdu = fits.PrimaryHDU(data=corrected_slices)
    
    # Access the FITS header
    header = hdu.header

    # Add metadata to the header
    header['NSlits'] = slices_shape[0]   # Number of slits
    header['NWavel'] = slices_shape[1]   # Number of wavelength points
    header['Nalpha'] = slices_shape[2]   # Number of alpha points
    header['PA_V3'] = PA_V3   # Position Angle (V3) in degrees
    header['TARG_RA'] = TARG_RA   # Target Right Ascension (in degrees)
    header['TARG_DEC'] = TARG_DEC   # Target Declination (in degrees)

    band = selected_chan[-1]
    if band == 'a':
        header['BAND'] = 'SHORT'
    elif band =='b':
        header['BAND'] = 'MEDIUM'
    elif band == 'c':
        header['BAND'] = 'LONG'
    else:
        raise NameError(f'Band name is not correct : {band}')

    # Create an HDUList to hold the primary HDU
    hdul = fits.HDUList([hdu])

    # Write the data and header to a new FITS file
    hdul.writeto(filename, overwrite=True)


def get_fits_target_coordinates(fits_path):
    """
    Extract target coordinates from FITS header.
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[1].header
        targ_ra = hdr['RA_V1']
        targ_dec = hdr['DEC_V1']
    return targ_ra, targ_dec

def get_fits_target_coordinates_corrected_data(fits_path):
    """
    Extract target coordinates from FITS header.
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        targ_ra = hdr['TARG_RA']
        targ_dec = hdr['TARG_DEC']
    return targ_ra, targ_dec


def get_data_from_fits(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    return data


def save_numpy_to_fits(data, metadata, filename, masks):
    """
    Sauvegarde un cube numpy (nw, ny, nx) en FITS
    avec des coordonnées linéaires pour les 3 axes.
    """

    unit_alpha = 'deg'
    unit_beta = 'deg'
    unit_wavelength = 'um'

    alpha_axis  = metadata['ALPHA_AXIS']
    beta_axis   = metadata['BETA_AXIS']
    wavelength  = metadata['WAVELENGTH']

    alpha_axis += metadata['RA_REF']
    beta_axis  += metadata['DEC_REF']

    hdu = fits.PrimaryHDU(data=data)
    header = hdu.header

    # data[0] = rotate(data[0], angle=metadata['PA_V3'], reshape=False)
    # data[1] = rotate(data[3], angle=-metadata['PA_V3'], reshape=False)
    # data[2] = rotate(data[1], angle=180-metadata['PA_V3'], reshape=False)
    # data[3] = rotate(data[2], angle=180-(90-metadata['PA_V3']), reshape=False)    
    # data[4] = rotate(data[4], angle=-(180-metadata['PA_V3']), reshape=False)
    # data[5] = rotate(data[5], angle=-(180-(90-metadata['PA_V3'])), reshape=False) 
    # data[6] = rotate(data[0], angle=-metadata['PA_V3'], reshape=False)
    # data[7]  = rotate(data[2], metadata['PA_V3']-360, reshape=False)
    # data[8]  = rotate(data[0], angle=metadata['PA_V3']-360-8.2, reshape=False)
    # data[9] = rotate(data[2], angle=metadata['PA_V3']-360+8.2, reshape=False)
    # data[10]  = rotate(data[1], angle=360-metadata['PA_V3'], reshape=False)
    # data[11] = rotate(data[2], angle=360-metadata['PA_V3']+8.2, reshape=False)
    # data[12] = rotate(data[1], angle=360-metadata['PA_V3']-8.2, reshape=False)
    # data[13] = rotate(data[0], angle=metadata['PA_V3']-180, reshape=False)
    # data[14] = np.fliplr(data[40])


    # data[15] = rotate(np.fliplr(data[40]), angle=metadata['PA_V3'], reshape=False)
    # data[16] = rotate(np.fliplr(data[40]), angle=-metadata['PA_V3'], reshape=False)
    # data[17] = rotate(np.fliplr(data[40]), angle=360-metadata['PA_V3'], reshape=False)
    # data[18] = rotate(np.fliplr(data[40]), angle=360-metadata['PA_V3']-8.2, reshape=False)    
    # data[19] = rotate(np.fliplr(data[40]), angle=360-metadata['PA_V3']+8.2, reshape=False)
    # data[20] = rotate(np.fliplr(data[40]), angle=metadata['PA_V3']-360, reshape=False)
    # data[21] = rotate(np.fliplr(data[40]), angle=metadata['PA_V3']-360-8.2, reshape=False)
    # data[22] = rotate(np.fliplr(data[40]), angle=metadata['PA_V3']-360+8.2, reshape=False)
    
    # data[23] = np.flipud(data[40])
    # data[24] = rotate(np.flipud(data[40]), angle=metadata['PA_V3'], reshape=False)
    # data[25] = rotate(np.flipud(data[40]), angle=-metadata['PA_V3'], reshape=False)
    # data[26] = rotate(np.flipud(data[40]), angle=360-metadata['PA_V3'], reshape=False)
    # data[27] = rotate(np.flipud(data[40]), angle=360-metadata['PA_V3']-8.2, reshape=False)    
    # data[28] = rotate(np.flipud(data[40]), angle=360-metadata['PA_V3']+8.2, reshape=False)
    # data[29] = rotate(np.flipud(data[40]), angle=metadata['PA_V3']-360, reshape=False)
    # data[30] = rotate(np.flipud(data[40]), angle=metadata['PA_V3']-360-8.2, reshape=False)
    # data[31] = rotate(np.flipud(data[40]), angle=metadata['PA_V3']-360+8.2, reshape=False)

    # data[32] = np.flipud(np.fliplr(data[40]))
    # data[33] = rotate(np.flipud(np.fliplr(data[40])), angle=metadata['PA_V3'], reshape=False)
    # data[34] = rotate(np.flipud(np.fliplr(data[40])), angle=-metadata['PA_V3'], reshape=False)
    # data[35] = rotate(np.flipud(np.fliplr(data[40])), angle=360-metadata['PA_V3'], reshape=False)
    # data[36] = rotate(np.flipud(np.fliplr(data[40])), angle=360-metadata['PA_V3']-8.2, reshape=False)    
    # data[37] = rotate(np.flipud(np.fliplr(data[40])), angle=360-metadata['PA_V3']+8.2, reshape=False)
    # data[38] = rotate(np.flipud(np.fliplr(data[40])), angle=metadata['PA_V3']-360, reshape=False)
    # data[39] = rotate(np.flipud(np.fliplr(data[40])), angle=metadata['PA_V3']-360-8.2, reshape=False)
    # data[40] = rotate(np.flipud(np.fliplr(data[40])), angle=metadata['PA_V3']-360+8.2, reshape=False)

    # for i in range(data.shape[0]):
    #     data[i] = rotate(np.flipud(np.fliplr(data[i])), angle=metadata['PA_V3']-360-8.2, reshape=False)


    # --- Métadonnées générales ---
    header['AUTHOR']   = 'Nicolas Monnier'
    header['NWAVEL']   = data.shape[0]   # Nombre de points spectraux
    header['NAXIS1']   = data.shape[2]   # taille en alpha
    header['NAXIS2']   = data.shape[1]   # taille en beta

    header['PA_V3']    = metadata['PA_V3']    # Position Angle (V3) in degrees
    header['TARG_RA']  = metadata['TARG_RA']   # Target Right Ascension (in degrees)
    header['TARG_DEC'] = metadata['TARG_DEC']  # Target Declination (in degrees)
    header['RA_V1']    = metadata['RA_V1']   # Target RA in V1 frame (in degrees)
    header['DEC_V1']   = metadata['DEC_V1']  # Target DEC in V1 frame (in degrees)
    header['RA_REF']   = metadata['RA_REF']   # Reference RA (in degrees)
    header['DEC_REF']  = metadata['DEC_REF']  # Reference DEC (in degrees)

    # --- Axe spatial X (alpha) ---
    da = np.mean(np.diff(alpha_axis))
    header['CTYPE1'] = 'ALPHA'
    header['CUNIT1'] = unit_alpha
    header['CRVAL1'] = alpha_axis[0]
    header['CDELT1'] = da
    header['CRPIX1'] = 1

    # --- Axe spatial Y (beta) ---
    db = np.mean(np.diff(beta_axis))
    header['CTYPE2'] = 'BETA'
    header['CUNIT2'] = unit_beta
    header['CRVAL2'] = beta_axis[0]
    header['CDELT2'] = db
    header['CRPIX2'] = 1

    # --- Axe spectral (approx linéaire) ---
    dw = np.mean(np.diff(wavelength))
    header['CTYPE3'] = 'WAVE'
    header['CUNIT3'] = unit_wavelength
    header['CRVAL3'] = wavelength[0]
    header['CDELT3'] = dw
    header['CRPIX3'] = 1

    # --- Table WCS-TABLE contenant le vecteur des longueurs d’onde ---
    col = fits.Column(name='wavelength', format=f'{len(wavelength)}E', dim=f'({len(wavelength)})', array=[wavelength])
    wcstable_hdu = fits.BinTableHDU.from_columns([col])
    wcstable_hdu.header['EXTNAME'] = 'WCS-TABLE'

    if masks is not None:
        masks = np.array(masks)
        mask_hdu = fits.ImageHDU(data=masks.astype(np.uint8), name='MASKS')

        # # --- Table MASKS-TABLE contenant les masques ---
        # col_masks = fits.Column(name='masks', format=f'{masks.shape[0]}L', dim=f'({masks.shape[0]})', array=[masks])
        # maskstable_hdu = fits.BinTableHDU.from_columns([col_masks])
        # maskstable_hdu.header['EXTNAME'] = 'MASKS-TABLE'
        # --- Créer la liste d'extensions ---
        hdul = fits.HDUList([hdu, wcstable_hdu, mask_hdu])
    else:
        # --- Créer la liste d'extensions ---
        hdul = fits.HDUList([hdu, wcstable_hdu])

    # --- Écriture ---
    hdul.writeto(filename, overwrite=True)
    print(f"✅ Fichier sauvegardé : {filename}")


def save_mirim_to_fits(data, filename, masks=None):
    """
    Sauvegarde un cube numpy (nw, ny, nx) en FITS
    avec des coordonnées linéaires pour les 3 axes.
    """

    unit_alpha = 'deg'
    unit_beta = 'deg'
    unit_wavelength = 'um'

    alpha_axis  = np.arange(data.shape[1]) #metadata['ALPHA_AXIS']
    beta_axis   = np.arange(data.shape[2]) # metadata['BETA_AXIS']
    wavelength  = np.arange(data.shape[0])

    alpha_axis += 0 # metadata['RA_REF']
    beta_axis  += 0 # metadata['DEC_REF']

    hdu = fits.PrimaryHDU(data=data)
    header = hdu.header

    # --- Métadonnées générales ---
    header['AUTHOR']   = 'Nicolas Monnier'
    header['NWAVEL']   = data.shape[0]   # Nombre de points spectraux
    header['NAXIS1']   = data.shape[2]   # taille en alpha
    header['NAXIS2']   = data.shape[1]   # taille en beta

    header['PA_V3']    = 0 #metadata['PA_V3']    # Position Angle (V3) in degrees
    header['TARG_RA']  = 0 #metadata['TARG_RA']   # Target Right Ascension (in degrees)
    header['TARG_DEC'] = 0 #metadata['TARG_DEC']  # Target Declination (in degrees)
    header['RA_V1']    = 0 #metadata['RA_V1']   # Target RA in V1 frame (in degrees)
    header['DEC_V1']   = 0 #metadata['DEC_V1']  # Target DEC in V1 frame (in degrees)
    header['RA_REF']   = 0 #metadata['RA_REF']   # Reference RA (in degrees)
    header['DEC_REF']  = 0 #metadata['DEC_REF']  # Reference DEC (in degrees)

    # --- Axe spatial X (alpha) ---
    da = np.mean(np.diff(alpha_axis))
    header['CTYPE1'] = 'ALPHA'
    header['CUNIT1'] = unit_alpha
    header['CRVAL1'] = alpha_axis[0]
    header['CDELT1'] = da
    header['CRPIX1'] = 1

    # --- Axe spatial Y (beta) ---
    db = np.mean(np.diff(beta_axis))
    header['CTYPE2'] = 'BETA'
    header['CUNIT2'] = unit_beta
    header['CRVAL2'] = beta_axis[0]
    header['CDELT2'] = db
    header['CRPIX2'] = 1

    # --- Axe spectral (approx linéaire) ---
    dw = np.mean(np.diff(wavelength))
    header['CTYPE3'] = 'WAVE'
    header['CUNIT3'] = unit_wavelength
    header['CRVAL3'] = wavelength[0]
    header['CDELT3'] = dw
    header['CRPIX3'] = 1

    # --- Table WCS-TABLE contenant le vecteur des longueurs d’onde ---
    col = fits.Column(name='wavelength', format=f'{len(wavelength)}E', dim=f'({len(wavelength)})', array=[wavelength])
    wcstable_hdu = fits.BinTableHDU.from_columns([col])
    wcstable_hdu.header['EXTNAME'] = 'WCS-TABLE'

    if masks is not None:
        masks = np.array(masks)
        mask_hdu = fits.ImageHDU(data=masks.astype(np.uint8), name='MASKS')

        # # --- Table MASKS-TABLE contenant les masques ---
        # col_masks = fits.Column(name='masks', format=f'{masks.shape[0]}L', dim=f'({masks.shape[0]})', array=[masks])
        # maskstable_hdu = fits.BinTableHDU.from_columns([col_masks])
        # maskstable_hdu.header['EXTNAME'] = 'MASKS-TABLE'
        # --- Créer la liste d'extensions ---
        hdul = fits.HDUList([hdu, wcstable_hdu, mask_hdu])
    else:
        # --- Créer la liste d'extensions ---
        hdul = fits.HDUList([hdu, wcstable_hdu])

    # --- Écriture ---
    hdul.writeto(filename, overwrite=True)
    print(f"✅ Fichier sauvegardé : {filename}")
