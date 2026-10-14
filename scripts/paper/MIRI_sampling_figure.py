import numpy as np
import matplotlib.pyplot as plt

# --- Données (à adapter avec valeurs exactes) ---
channels = {
    "Ch1": {"lam_min": 4.9, "lam_max": 7.7, "pixel": 0.196, "slit": 0.176},
    "Ch2": {"lam_min": 7.7, "lam_max": 11.9, "pixel": 0.245, "slit": 0.277},
    "Ch3": {"lam_min": 11.9, "lam_max": 18.3, "pixel": 0.273, "slit": 0.387},
    "Ch4": {"lam_min": 18.3, "lam_max": 28.1, "pixel": 0.387, "slit": 0.645},
}

# --- Diffraction ---
D = 6.5
lam = np.linspace(5, 28, 500) * 1e-6

theta = 1.22 * lam / D * 206265

# --- Plot ---
plt.figure(figsize=(7,4.5))

# Diffraction
plt.plot(lam*1e6, theta, 'r-', label='Diffraction limit')
plt.plot(lam*1e6, theta/2, 'k--', label='Ideal sampling')


# Colors (fixes)
color_pixel = "cornflowerblue"
color_slit = "sandybrown"

for ch, vals in channels.items():

    # Slit width (bleu)
    plt.hlines(vals["slit"],
               vals["lam_min"],
               vals["lam_max"],
               colors=color_slit,
               linestyles="--",
               linewidth=2.5)

    # Pixel size (rouge)
    plt.hlines(vals["pixel"],
               vals["lam_min"],
               vals["lam_max"],
               colors=color_pixel,
               linestyles="--",
               linewidth=2.5)

    # Position du label au centre du segment
    x_mid = 0.5 * (vals["lam_min"] + vals["lam_max"])

    # Label légèrement au-dessus du pixel
    plt.text(x_mid,
             vals["pixel"] + 0.015,
             ch,
             ha='center',
             fontsize=14)


imager_sampling = 0.11
plt.hlines(imager_sampling, 5, 28, colors='m', linestyles=':', linewidth=1.5, label='Imager sampling')


# Légende manuelle (plus propre)
plt.plot([], [], color=color_slit, linewidth=3, label='Slit width')
plt.plot([], [], color=color_pixel, linewidth=3, label='Pixel size')

# Axes
plt.xlabel("Wavelength (µm)", fontsize=16)
plt.ylabel("Size (arcsec)", fontsize=16)
plt.xlim(5, 28)
plt.ylim(0.0, 1.1)

#Increase ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/nmonnier/Presentations/Fusion_figures/MIRI_sampling_figure.png", dpi=300)  # Sauvegarde haute résolution
plt.show()