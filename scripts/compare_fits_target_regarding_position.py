import numpy 
import pickle
import math
from pathlib import Path
import os
from astropy.io import fits

fits_dir = ''
dir_ = Path('/data/cluster/JWST/RealData/01192/NGC_7023/MRS/Science/run_2024_06/Science/stage0')

save_dit = ''


os.chdir(str(dir_))
dictio = {}

path_exists = Path.exists(dir_ + 'saved_dict.pkl')

if not path_exists:
    for pth in dir_.iterdir():
        hdul = fits.open(pth)
        hdr = hdul[1].header
        targ_ra  = hdr['RA_V1']
        targ_dec = hdr['DEC_V1']
        dictio[pth] = {}
        dictio[pth]['targ_ra'] = targ_ra
        dictio[pth]['targ_dec'] = targ_dec

    with open(save_dit + 'saved_dict.pkl', 'wb') as f:
        pickle.dump(dictio, f)



# with open('saved_dictB.pkl', 'rb') as f:
#    loaded_dictB = pickle.load(f)
#    with open('saved_dictA.pkl', 'rb') as f:
#        loaded_dictA = pickle.load(f)
#    with open('saved_dictC.pkl', 'rb') as f:
#         loaded_dictC = pickle.load(f)

# for i in loaded_dictA:
#    if 'w01288002001_0210f_00001_mirifushort' in str(i):
#         print(i)
#         ref_dec = loaded_dictA[i]['targ_dec']
#         ref_ra = loaded_dictA[i]['targ_ra']

with open('saved_dict.pkl', 'rb') as f:
    loaded_dictC = pickle.load(f)

closedist = 100000000
name = ''
ref_ra = 315.282905
ref_dec = 68.173472
for n_file in range(len(loaded_dictC)):
    for i in loaded_dictC:
        dec = loaded_dictC[i]['targ_dec']
        ra = loaded_dictC[i]['targ_ra']
        dist = math.sqrt((dec-ref_dec)**2 + (ra-ref_ra)**2)
        if dist < closedist:
            closedist = dist
            name = i
    print(name)
    del loaded_dictC[name]
    

