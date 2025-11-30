import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

# Load data
# ref_cube = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/fusion_mrs_simulated_cube.npy')
templates = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_templates.npy')
maps = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/simulation_maps.npy')
wavelength = np.load('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Templates/wavel_axis_NGC7023_1ABC_2ABC_3ABC_4AB.npy')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Données
n = 4
maps4 = maps[:n]
templates4 = templates[:n]

# Styles N&B : linestyles seulement
linestyles = ["-", "--", "-.", ":"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# Figure large mais compacte
fig = plt.figure(figsize=(11, 5))

# ---------------------------------------------------------
# 1) Ligne du haut : 4 MAPS
# ---------------------------------------------------------
for i in range(n):
    ax = fig.add_subplot(2, n, i+1)
    im = ax.imshow(maps4[i], origin="lower", cmap="viridis")
    ax.set_title(f"Map {i+1}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar fine collée à droite
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)

# ---------------------------------------------------------
# 2) Ligne du bas : Templates (4 courbes)
# ---------------------------------------------------------
ax_spec = fig.add_subplot(2, 1, 2)  # 2ème ligne entière

for i in range(n):
    ax_spec.plot(
        wavelength,
        templates4[i],
        lw=1.2,
        ls=linestyles[i],
        label=f"Template {i+1}"
    )

ax_spec.set_xlabel(r"Wavelength ($\mu$m)")
ax_spec.set_ylabel("Flux (arbitrary units)")
ax_spec.legend(frameon=True, fontsize=9, ncol=2)
ax_spec.grid(True, linestyle="--", alpha=0.4)  # grille pour lecture facile

plt.tight_layout()
# plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_ground_truth.png', dpi=300)
# plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_ground_truth.pdf')
plt.show()

