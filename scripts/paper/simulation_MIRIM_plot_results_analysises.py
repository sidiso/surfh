import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt



# Load data
ref_cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
templates = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_templates.npy')
wavelength = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy')

# for i in range(templates.shape[0]):
#     print(f'Wavelength where template is not zero for template {i}: {np.where(templates[i,:]!=0)[0]}')
spectral_line_idx = [937, 1740, 2678, 3778, 5077, 6619, 7537, 8435]

# Load results data
res_dir = '/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Results/MIRIM_lcg_Temp_13_nit_500_mu_0.00e+00_SD_True/res_cube.fits'

with fits.open(res_dir) as hdul:
    res_cube = hdul[0].data

print("SHape cube = ", res_cube.shape)
print("SHape wavel = ", wavelength.shape)
print("Template shape = ", templates.shape)

continum_masked_res_cube = res_cube.copy()
continum_masked_ref_cube = ref_cube.copy()

for sline_idx in spectral_line_idx:
    continum_masked_res_cube[sline_idx,:,:] = (continum_masked_res_cube[sline_idx-1,:,:] + continum_masked_res_cube[sline_idx+1,:,:])/2
    continum_masked_ref_cube[sline_idx,:,:] = (continum_masked_ref_cube[sline_idx-1,:,:] + continum_masked_ref_cube[sline_idx+1,:,:])/2

mean_masked_res_cube = np.nanmean(res_cube, axis=(1,2))
mean_masked_ref_cube = np.nanmean(ref_cube, axis=(1,2))

mean_continuum_masked_res_cube = np.nanmean(continum_masked_res_cube, axis=(1,2))
mean_continuum_masked_ref_cube = np.nanmean(continum_masked_ref_cube, axis=(1,2))

rel_error = (mean_continuum_masked_res_cube - mean_continuum_masked_ref_cube) / mean_continuum_masked_ref_cube

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.0,
})

# Figure large — idéal pour A&A
fig, axs = plt.subplots(
    2, 1,
    figsize=(12, 6),
    sharex=True,
    gridspec_kw={"hspace": 0.15}
)

# ---------------------------------------------------
# 1) PANEL DU HAUT : Erreur relative
# ---------------------------------------------------
axs[0].plot(wavelength[:-50], rel_error[:-50], lw=1.2)
axs[0].set_ylabel(r"Relative error")
# axs[0].set_title("Relative error on the reconstructed continuum")

# Grille
axs[0].grid(True, linestyle="--", alpha=0.4)

# ---------------------------------------------------
# 2) PANEL DU BAS : Signal moyen des deux cubes
# ---------------------------------------------------
axs[1].plot(wavelength[:-50], mean_masked_res_cube[:-50],
            label="Reference cube", lw=1.3)
axs[1].plot(wavelength[:-50], mean_masked_ref_cube[:-50],
            label="Reconstructed cube", lw=1.3)

axs[1].set_xlabel(r"Wavelength ($\mu$m)")
axs[1].set_ylabel(r"Flux (MJy sr$^{-1}$)")

# Grille
axs[1].grid(True, linestyle="--", alpha=0.4)

# Légende avec un bord (A&A style)
axs[1].legend(
    loc="upper left",
    frameon=True,         # ← bord activé
    framealpha=1.0,       # bord opaque
    edgecolor="black",    # bord noir fin
    fontsize=10
)
# plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_flux_error.png', dpi=300)
# plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_flux_error.pdf')
plt.show()