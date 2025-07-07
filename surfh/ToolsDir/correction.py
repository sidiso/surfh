import numpy as np
from scipy.ndimage import generic_filter


def interpolate_negatives_3d(cube):
    cube = cube.copy()  # Pour ne pas modifier l'original

    def interpolate_func(values):
        center = values[len(values) // 2]
        if center >= 0:
            return center
        else:
            # Ne garder que les valeurs positives autour
            positives = values[values >= 0]
            return np.mean(positives) if len(positives) > 0 else 0.0

    # Appliquer un filtre 3D (voisinage 3x3x3)
    result = generic_filter(cube, interpolate_func, size=3, mode='mirror')
    return result


def interpolate_negatives_2d(image):
    image = image.copy()

    def interpolate_func(values):
        center = values[len(values) // 2]
        if center >= 0 and not np.isnan(center):
            return center
        # voisins positifs non NaN
        positives = values[(values >= 0)]
        return np.nanmean(positives) if positives.size > 0 else 0.0

    # Filtre 3x3 pour interpoler chaque pixel avec ses voisins
    result = generic_filter(image, interpolate_func, size=3, mode='mirror')
    return result
