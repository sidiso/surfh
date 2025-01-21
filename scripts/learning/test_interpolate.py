#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.misc
from matplotlib import transforms

from surfh.Models import smallmiri as mrs

im = scipy.misc.ascent()

xa = (
    np.linspace(
        -mrs.ch4a.fov.alpha_width / 2, mrs.ch4a.fov.alpha_width / 2, im.shape[0]
    )
    + mrs.ch4a.fov.origin.alpha
)


xb = (
    np.linspace(
        -mrs.ch4a.fov.beta_width / 2, mrs.ch4a.fov.beta_width / 2, im.shape[1]
    )
    + mrs.ch4a.fov.origin.beta
)

step = np.mean(np.diff(xa))
fov = mrs.ch1a.fov
fov.angle = 8
na, nb = fov.coords(step)

xo = np.vstack(
    [
        np.tile(xa.reshape((-1, 1)), (1, len(xb))).ravel(),
        np.tile(xb.reshape((1, -1)), (len(xa), 1)).ravel(),
    ]
).T

xi = np.vstack(
    [
        np.asarray(na).ravel(),
        np.asarray(nb).ravel(),
    ]
).T


"""
L'image est centrée sur le FoV du ch4a en partageant les mêmes coordonées. 
Un deuxième FoV basé sur ch1a est initialisée. FoV plus petit et avec un angle de 8°.
scipy.interpolate permet de faire l'interpolation du FoV de ch1a sur le FoV du ch4a pour observé l'image sur une grille 2D.
Dans scipy.interpolate.interpn :    xa , xb) -> Coordonnées de l'image d'entrée
                                    im       -> Valeurs associées aux coordonées
                                    xi       -> Coordonées à interpolée pour l'image de sortie
                                    nim      -> Grille 2D de l'image de sortie interpolée
"""
nim = scipy.interpolate.interpn((xi, xb), im, xi).reshape(na.shape)

plt.figure(1)
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(im, extent=(xb[0], xb[-1], xa[-1], xa[0]))
plt.plot(xi[:, 1], xi[:, 0], ".", color="black", alpha=0.1)
plt.plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    color="white",
)
plt.subplot(2, 2, 2)
plt.imshow(nim)
plt.subplot(2, 2, 3)
plt.imshow(
    nim,
    transform=transforms.Affine2D().rotate_deg(-fov.angle) + plt.gca().transData,
)
plt.subplot(2, 2, 4)
plt.plot(xo[:, 0], xo[:, 1], ".", alpha=0.2)
plt.plot(xi[:, 0], xi[:, 1], ".", alpha=0.2)

# sys.exit(0)

#%% Cube \

im2 = scipy.misc.face()[-im.shape[0] :, -im.shape[1] :, 0]
cube = np.moveaxis(np.dstack([im, im2]), 2, 0)
z = np.array([0, 1])

npix = na.size
nl = z.size

xi = np.vstack(
    [
        np.tile(z.reshape((-1, 1, 1)), (1, na.shape[0], na.shape[1])).ravel(),
        np.tile(na[np.newaxis], (nl, 1, 1)).ravel(),
        np.tile(nb[np.newaxis], (nl, 1, 1)).ravel(),
    ]
).T


""""
Même chose que précedemment mais avec un cube d'image
"""
ncube = scipy.interpolate.interpn((z, xa, xb), cube, xi).reshape((nl,) + na.shape)

plt.figure(2)
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(cube[0], extent=(xb[0], xb[-1], xa[-1], xa[0]))
plt.plot(nb, na, ".", color="black", alpha=0.005)
plt.plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    color="white",
)
plt.subplot(2, 2, 2)
plt.imshow(cube[1], extent=(xb[0], xb[-1], xa[-1], xa[0]))
plt.plot(nb, na, ".", color="black", alpha=0.005)
plt.plot(
    [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    "-o",
    color="white",
)
plt.subplot(2, 2, 3)
plt.imshow(ncube[0])
plt.subplot(2, 2, 4)
plt.imshow(ncube[1])

plt.show()