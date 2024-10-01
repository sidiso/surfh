#!/usr/bin/env python3

"""Quickly test of surfh"""

import operator as op

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from surfh.Models import instru

c1 = instru.Coord(5, 5)

f = instru.FOV(2, 3)
a, b = f.local2global(*f.local_coords(0.23, alpha_margin=0, beta_margin=0))

plt.figure(1)
plt.clf()
plt.plot(a, b, ".", color="k")
plt.plot(f.origin.alpha, f.origin.beta, "o")
plt.plot(
    list(map(op.attrgetter("alpha"), f.vertices)) + [f.vertices[0].alpha],
    list(map(op.attrgetter("beta"), f.vertices)) + [f.vertices[0].beta],
    "-x",
)



f = instru.FOV(2, 3, angle=-10, origin=c1)
f.shift(instru.Coord(-3, -3))
f = f + instru.Coord(1, 1)
f.rotate(0)
a, b = f.local2global(*f.local_coords(0.23, alpha_margin=0.5, beta_margin=-0.5))
plt.plot(a, b, ".", color="b")
plt.plot(f.origin.alpha, f.origin.beta, "o")
plt.plot(
    list(map(op.attrgetter("alpha"), f.vertices)) + [f.vertices[0].alpha],
    list(map(op.attrgetter("beta"), f.vertices)) + [f.vertices[0].beta],
    "-o",
)
plt.plot(
    [
        f.bbox[0].alpha,
        f.bbox[1].alpha,
        f.bbox[1].alpha,
        f.bbox[0].alpha,
        f.bbox[0].alpha,
    ],
    [f.bbox[0].beta, f.bbox[0].beta, f.bbox[1].beta, f.bbox[1].beta, f.bbox[0].beta],
    "-o",
)



plt.axis("equal")
plt.show()