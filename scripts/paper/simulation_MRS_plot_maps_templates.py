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







# --- MAP À ZOOMER (la 2e) ---
map2 = maps4[1]

# --- ZONE DE ZOOM (à adapter) ---
# Par exemple : zoom sur x=30→80 et y=40→100
x1, x2 = 24, 51
y1, y2 = 24, 60

zoomed = map2[y1:y2, x1:x2]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# im = ax.imshow(zoomed, origin="lower", cmap="viridis")
# ax.set_title("Zoom on Map 2")

# # ticks optionnels pour lecture
# ax.set_xticks([])
# ax.set_yticks([])

# # colorbar fine
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="4%", pad=0.05)
# fig.colorbar(im, cax=cax)

# plt.tight_layout()




fig = plt.figure(figsize=(9, 4))

# --- panel 1 : map originale ---
ax1 = fig.add_subplot(1, 2, 1)
im1 = ax1.imshow(map2, origin='lower', cmap='viridis')
ax1.set_title("Map 2 (full)")
ax1.set_xticks([]); ax1.set_yticks([])

# rectangle du zoom
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                     linewidth=1, edgecolor='red', facecolor='none')
ax1.add_patch(rect)

# colorbar fine
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.02)
fig.colorbar(im1, cax=cax1)

# --- panel 2 : zone zoomée ---
ax2 = fig.add_subplot(1, 2, 2)
im2 = ax2.imshow(zoomed, origin='lower', cmap='viridis')
ax2.set_title("Zoomed region")
ax2.set_xticks([]); ax2.set_yticks([])

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.02)
fig.colorbar(im2, cax=cax2)

plt.tight_layout()
# plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_zoom_ground_truth.png', dpi=300)
# plt.savefig('/home/nmonnier/Data/JWST/Simulation/Paper/Fusion/Plots/simulation_zoom_ground_truth.pdf')
plt.show()

