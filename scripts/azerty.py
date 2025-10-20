import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA

# -----------------------------
# 1. Génération de données simulées
# -----------------------------

np.random.seed(0)

# Dimensions : 100 spectres de 50 canaux chacun
n_samples, n_features = 100, 50

# Deux composantes spectrales "physiques"
t = np.linspace(0, 1, n_features)
comp1 = np.exp(-50 * (t - 0.3)**2)         # pic centré à 0.3
comp2 = np.exp(-40 * (t - 0.7)**2) * 0.8   # pic centré à 0.7

# Mélange linéaire aléatoire de ces deux composantes
weights = np.random.rand(n_samples, 2)
true_data = weights @ np.vstack([comp1, comp2])

# Ajout de bruit centré (positif/négatif)
noise = 0.05 * np.random.randn(*true_data.shape)
X = true_data + noise

print(f"Min: {X.min():.3f}, Max: {X.max():.3f}")

# -----------------------------
# 2. Méthode 1 : NMF classique (nécessite X >= 0)
# -----------------------------

X_nmf = X.copy()
X_nmf[X_nmf < 0] = 0  # troncature

nmf = NMF(n_components=4, init='nndsvda', random_state=0, max_iter=1000)
W_nmf = nmf.fit_transform(X_nmf)
H_nmf = nmf.components_
recon_nmf = W_nmf @ H_nmf

# -----------------------------
# 3. Méthode 2 : PCA (accepte les valeurs négatives)
# -----------------------------

pca = PCA(n_components=4)
W_pca = pca.fit_transform(X)
H_pca = pca.components_
recon_pca = W_pca @ H_pca + pca.mean_

# -----------------------------
# 4. Méthode 3 : Offset / Semi-NMF (approche empirique)
# -----------------------------

offset = -X.min() + 1e-6
X_shifted = X + offset

nmf_off = NMF(n_components=4, init='nndsvda', random_state=0, max_iter=1000)
W_off = nmf_off.fit_transform(X_shifted)
H_off = nmf_off.components_
recon_off = W_off @ H_off - offset

# -----------------------------
# 5. Visualisation
# -----------------------------

fig, axs = plt.subplots(4, 2, figsize=(10, 8))
axs = axs.ravel()

# Données réelles
axs[0].plot(t, comp1, 'r--', label="True comp1")
axs[0].plot(t, comp2, 'b--', label="True comp2")
axs[0].set_title("Composantes réelles")
axs[0].legend()

# NMF
axs[1].plot(t, H_nmf[0], 'r', label="NMF comp1")
axs[1].plot(t, H_nmf[1], 'b', label="NMF comp2")
axs[1].set_title("NMF (négatifs tronqués)")
axs[1].legend()

# PCA
for i in range(H_pca.shape[0]):
    axs[2].plot(t, H_pca[i], 'r', label=f"PCA comp{i}")
axs[2].set_title("PCA (valeurs signées)")
axs[2].legend()

# Offset NMF
for i in range(H_off.shape[0]):
    axs[3].plot(t, H_off[i], 'r', label=f"Offset comp{i}")
axs[3].set_title("Offset / Semi-NMF")
axs[3].legend()

# Comparaison reconstruction
axs[4].imshow(X, aspect='auto', cmap='RdBu', vmin=-0.2, vmax=0.5)
axs[4].set_title("Signal original")

axs[5].imshow(recon_off, aspect='auto', cmap='RdBu', vmin=-0.2, vmax=0.5)
axs[5].set_title("Reconstruction (Offset NMF)")

axs[6].imshow(X-recon_off, aspect='auto', cmap='RdBu', vmin=-0.2, vmax=0.5)
axs[6].set_title("Diff")


plt.tight_layout()
plt.show()
