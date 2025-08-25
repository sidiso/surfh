import numpy as np
import os
from rich import print
from astropy.io import fits


def extract_name_raw(dir):
    keywords = dir.split('_')
    return keywords[0], keywords[1], keywords[2], keywords[3]

def extract_name_corr_filt(dir):
    keywords = dir.split('_')
    return keywords[0], keywords[1]


dir = '/home/nmonnier/Data/JWST/small_NGC/Fusion/'
# dir = '/home/nmonnier/Data/JWST/Point_source/Fusion/'
raw_slices_dir = dir + 'Raw_slices/'
corrected_slices_dir = dir + 'Corrected_slices/'
filtered_slices_dir = dir + 'Filtered_slices/'

corrected_files = os.listdir(corrected_slices_dir)
filtered_files = os.listdir(filtered_slices_dir)

for file in os.listdir(raw_slices_dir):
    
    ch1, ch2, obs, dith = extract_name_raw(file)
    
    match_files = [t for t in corrected_files if ch1 in t]
    if any(match_files):
        for slice_file in sorted(match_files):
            if(dith in slice_file):
                print(f'Writting metadata from {file} to {slice_file}')
                raw_hdulist = fits.open(raw_slices_dir+file)
                header_glob = raw_hdulist[0].header
                xoffset = header_glob['XOFFSET'] / 3600
                yoffset = header_glob['YOFFSET'] / 3600
                header_sci = raw_hdulist[1].header
                new_ra = header_sci['RA_V1']#/3600
                new_dec = header_sci['DEC_V1']#/3600
                new_ra_v1 = header_sci['RA_V1']
                new_dec_v1 = header_sci['DEC_V1']
                new_ra_ref = header_sci['RA_REF']
                new_dec_ref = header_sci['DEC_REF']
                raw_hdulist.close()

                corrected_hdulist = fits.open(corrected_slices_dir+slice_file)
                header = corrected_hdulist[0].header
                header['TARG_RA'] = new_ra
                header['TARG_DEC'] = new_dec
                header['XOFFSET'] = xoffset
                header['YOFFSET'] = yoffset
                header['RA_V1']   = (new_ra_v1, '(deg) RA of telescope V1 axis')
                header['DEC_V1']  = (new_dec_v1, '(deg) Dec of telescope V1 axis')
                header['RA_REF']  = (new_ra_ref, '(deg) Right Ascension of the reference point')
                header['DEC_REF'] = (new_dec_ref, '(deg) Declination of the reference point')
                corrected_hdulist.writeto(corrected_slices_dir+slice_file, overwrite=True)
                corrected_hdulist.close()

    match_files = [t for t in corrected_files if ch2 in t]
    if any(match_files):
        for slice_file in sorted(match_files):
            if(dith in slice_file):
                print(f'Writting metadata from {file} to {slice_file}')
                raw_hdulist = fits.open(raw_slices_dir+file)
                header_glob = raw_hdulist[0].header
                xoffset = header_glob['XOFFSET'] / 3600
                yoffset = header_glob['YOFFSET'] / 3600
                header_sci = raw_hdulist[1].header
                new_ra = header_sci['RA_V1']#/3600
                new_dec = header_sci['DEC_V1']#/3600
                new_ra_v1 = header_sci['RA_V1']
                new_dec_v1 = header_sci['DEC_V1']
                new_ra_ref = header_sci['RA_REF']
                new_dec_ref = header_sci['DEC_REF']
                raw_hdulist.close()

                corrected_hdulist = fits.open(corrected_slices_dir+slice_file)
                header = corrected_hdulist[0].header
                header['TARG_RA'] = new_ra
                header['TARG_DEC'] = new_dec
                header['XOFFSET'] = xoffset
                header['YOFFSET'] = yoffset
                header['RA_V1']   = (new_ra_v1, '(deg) RA of telescope V1 axis')
                header['DEC_V1']  = (new_dec_v1, '(deg) Dec of telescope V1 axis')
                header['RA_REF']  = (new_ra_ref, '(deg) Right Ascension of the reference point')
                header['DEC_REF'] = (new_dec_ref, '(deg) Declination of the reference point')
                corrected_hdulist.writeto(corrected_slices_dir+slice_file, overwrite=True)
                corrected_hdulist.close()

    match_files = [t for t in filtered_files if ch1 in t]
    if any(match_files):
        for slice_file in sorted(match_files):
            if(dith in slice_file):
                print(f'Writting metadata from {file} to {slice_file}')
                raw_hdulist = fits.open(raw_slices_dir+file)
                header_glob = raw_hdulist[0].header
                xoffset = header_glob['XOFFSET'] / 3600
                yoffset = header_glob['YOFFSET'] / 3600
                header_sci = raw_hdulist[1].header
                new_ra = header_sci['RA_V1']#/3600
                new_dec = header_sci['DEC_V1']#/3600
                new_ra_v1 = header_sci['RA_V1']
                new_dec_v1 = header_sci['DEC_V1']
                new_ra_ref = header_sci['RA_REF']
                new_dec_ref = header_sci['DEC_REF']
                raw_hdulist.close()

                corrected_hdulist = fits.open(filtered_slices_dir+slice_file)
                header = corrected_hdulist[0].header
                header['TARG_RA'] = new_ra
                header['TARG_DEC'] = new_dec
                header['XOFFSET'] = xoffset
                header['YOFFSET'] = yoffset
                header['RA_V1']   = (new_ra_v1, '(deg) RA of telescope V1 axis')
                header['DEC_V1']  = (new_dec_v1, '(deg) Dec of telescope V1 axis')
                header['RA_REF']  = (new_ra_ref, '(deg) Right Ascension of the reference point')
                header['DEC_REF'] = (new_dec_ref, '(deg) Declination of the reference point')
                corrected_hdulist.writeto(filtered_slices_dir+slice_file, overwrite=True)
                corrected_hdulist.close()

    match_files = [t for t in filtered_files if ch2 in t]
    if any(match_files):
        for slice_file in sorted(match_files):
            if(dith in slice_file):
                print(f'Writting metadata from {file} to {slice_file}')
                raw_hdulist = fits.open(raw_slices_dir+file)
                header_glob = raw_hdulist[0].header
                xoffset = header_glob['XOFFSET'] / 3600
                yoffset = header_glob['YOFFSET'] / 3600
                header_sci = raw_hdulist[1].header
                new_ra = header_sci['RA_V1']#/3600
                new_dec = header_sci['DEC_V1']#/3600
                new_ra_v1 = header_sci['RA_V1']
                new_dec_v1 = header_sci['DEC_V1']
                new_ra_ref = header_sci['RA_REF']
                new_dec_ref = header_sci['DEC_REF']
                raw_hdulist.close()

                corrected_hdulist = fits.open(filtered_slices_dir+slice_file)
                header = corrected_hdulist[0].header
                header['TARG_RA'] = new_ra
                header['TARG_DEC'] = new_dec
                header['XOFFSET'] = xoffset
                header['YOFFSET'] = yoffset
                header['RA_V1']   = (new_ra_v1, '(deg) RA of telescope V1 axis')
                header['DEC_V1']  = (new_dec_v1, '(deg) Dec of telescope V1 axis')
                header['RA_REF']  = (new_ra_ref, '(deg) Right Ascension of the reference point')
                header['DEC_REF'] = (new_dec_ref, '(deg) Declination of the reference point')
                corrected_hdulist.writeto(filtered_slices_dir+slice_file, overwrite=True)
                corrected_hdulist.close()
