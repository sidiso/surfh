import glob
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from reproject import reproject_interp
import numpy as np
from multiprocessing import Pool, cpu_count

from surfh.Vizualisation.cube_vizualisation import plot_cube


def get_wavelength_fits(hdul):
    # Return wavelengths axis from fits file
    CRVAL3 = hdul[1].header['CRVAL3']
    CRPIX3 = hdul[1].header['CRPIX3']
    CDELT3 = hdul[1].header['CDELT3']
    i = np.arange(hdul[1].data.shape[0])
    return CRVAL3 + (i + 1 - CRPIX3) * CDELT3


def compute_global_footprint(wcs_list, shapes):
    """Calcule l'enveloppe spatiale couvrant tous les cubes"""

    ras = []
    decs = []

    for wcs, shape in zip(wcs_list, shapes):
        ny, nx = shape
        corners = np.array([
            [0, 0],
            [0, ny],
            [nx, 0],
            [nx, ny]
        ])
        sky = wcs.wcs_pix2world(corners, 0)
        ras.extend(sky[:, 0])
        decs.extend(sky[:, 1])

    return min(ras), max(ras), min(decs), max(decs)


def make_target_wcs(ra_min, ra_max, dec_min, dec_max, pixscale):

    ra_center = 0.5 * (ra_min + ra_max)
    dec_center = 0.5 * (dec_min + dec_max)

    nx = int(np.ceil((ra_max - ra_min) / pixscale))
    ny = int(np.ceil((dec_max - dec_min) / pixscale))

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-pixscale, pixscale]

    return w, (ny, nx)

def reproject_cube_spatial(data, wcs3d, target_wcs2d, shape2d):

    nz = data.shape[0]
    out = np.full((nz, shape2d[0], shape2d[1]), np.nan, dtype=np.float32)

    wcs2d = wcs3d.dropaxis(2)
    print("Reproject")
    for k in range(nz):
        if k % 50 == 0:
            print(f"Slice n°{k+1} / {nz}")
        reproj, _ = reproject_interp(
            (data[k], wcs2d),
            target_wcs2d,
            shape_out=shape2d
        )
        out[k] = reproj

    return out


def _reproject_one(args):
    slice2d, wcs2d, target_wcs2d, shape2d, k = args
    reproj, _ = reproject_interp(
        (slice2d, wcs2d),
        target_wcs2d,
        shape_out=shape2d
    )
    return k, reproj


def reproject_cube_spatial_parallel(data, wcs3d, target_wcs2d, shape2d, nproc):

    nz = data.shape[0]
    out = np.full((nz, shape2d[0], shape2d[1]), np.nan, dtype=np.float32)

    wcs2d = wcs3d.dropaxis(2)

    args = [
        (data[k], wcs2d, target_wcs2d, shape2d, k)
        for k in range(nz)
    ]

    with Pool(processes=nproc) as pool:
        for k, reproj in pool.imap_unordered(_reproject_one, args, chunksize=4):
            out[k] = reproj

    return out


cubes_dir = '/home/nmonnier/Data/JWST/NGC_7023/Scan/Full_Scan/Channels/'
output_cube = cubes_dir + 'merged_cube.fits'

pixel_resolution = 0.1  # arcsec/pixel
pixel_resolution_deg = pixel_resolution / 3600.0  # degrees/pixel

cube_files = sorted(glob.glob(cubes_dir + "/*.fits"))
print(cube_files)


cubes = []

# Get data and metada for each cube
for f in cube_files:
    print("Gettting information from ", f)
    hdul = fits.open(f)
    # print(hdul[1].header)

    data = hdul[1].data.astype(np.float32) 
    wcs3d = WCS(hdul[1].header)

    wave = get_wavelength_fits(hdul)

    cubes.append({
        "file": f,
        "data": data,
        "wcs": wcs3d,
        "wave": wave,
        "shape_2d": data.shape[1:]
    })

    hdul.close()

print(len(cubes), "cubes loaded.")



wcs2d_list = [c["wcs"].dropaxis(2) for c in cubes]
shapes = [c["shape_2d"] for c in cubes]

ra_min, ra_max, dec_min, dec_max = compute_global_footprint(wcs2d_list, shapes)

target_wcs2d, target_shape2d = make_target_wcs(
    ra_min, ra_max, dec_min, dec_max, pixel_resolution_deg
)

print("Taille finale spatiale:", target_shape2d)
NPROC = cpu_count()-2
for c in cubes:
    print(" ", c["file"])
    c["data"] = reproject_cube_spatial_parallel(
        c["data"], c["wcs"], target_wcs2d, target_shape2d, NPROC
    )


for c in cubes:
    c["lmin"] = c["wave"].min()
    c["lmax"] = c["wave"].max()

cubes = sorted(cubes, key=lambda x: x["lmin"])

final_data = []
final_wave = []

for i, c in enumerate(cubes):

    w = c["wave"]
    d = c["data"]

    # Calcul des indices à garder. On garde uniquement les canaux qui se chevauchent de la bande sectrale du cube suivant (ex ch1a et ch1b, on garde le chevauchement du ch1b)
    if i < len(cubes) - 1:
        next_lmin = cubes[i+1]["lmin"]
        keep = w < next_lmin
    else:
        keep = np.ones_like(w, dtype=bool)

    if np.any(keep):
        final_data.append(d[keep])
        final_wave.append(w[keep])
        print(f"  {c['file']} : conservé {keep.sum()} / {len(w)} canaux")
    else:
        print(f"  {c['file']} : aucun canal conservé")




final_cube = np.concatenate(final_data, axis=0)
final_wavel = np.concatenate(final_wave)

print("Cube final:", final_cube.shape)


plot_cube(final_cube, final_wavel)

""""
Ecriture du cube final Dans un fichier Fits.
Attention : le header et les données changent par rapport aux fits initiaux.
Attention : La longueur d'onde n'est pas bien lu par DS9.
"""

print("Écriture du fichier final:", output_cube)

hdr = target_wcs2d.to_header()

hdr["NAXIS"]  = 3
hdr["NAXIS1"] = final_cube.shape[2]
hdr["NAXIS2"] = final_cube.shape[1]
hdr["NAXIS3"] = final_cube.shape[0]

# Axe spectral tabulé
hdr["CTYPE3"] = "WAVE-TAB"
hdr["CUNIT3"] = "um"
hdr["PS3_0"] = "WCS-TAB"
hdr["PS3_1"] = "WAVELENGTH"
hdr["COMMENT"] = "Spectral axis defined by WCS-TAB and WAVELENGTH extension"

hdu_cube = fits.PrimaryHDU(final_cube.astype(np.float32), header=hdr)

# Table WCS-TAB officielle
tab = Table()
tab["WAVELENGTH"] = final_wavel.astype(np.float64)
hdu_wcstab = fits.BinTableHDU(tab, name="WCS-TAB")
hdu_wcstab.header["TTYPE1"] = "WAVELENGTH"
hdu_wcstab.header["TUNIT1"] = "um"

# Extension image simple
hdu_wave = fits.ImageHDU(final_wavel.astype(np.float64), name="WAVELENGTH")
hdu_wave.header["BUNIT"] = "um"
hdu_wave.header["COMMENT"] = "Wavelength of slice i = WAVELENGTH[i]"

fits.HDUList([hdu_cube, hdu_wcstab, hdu_wave]).writeto(output_cube, overwrite=True)

print("Terminé ✅")