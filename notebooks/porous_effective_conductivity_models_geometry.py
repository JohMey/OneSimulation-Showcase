"""Geometry of the OneSimulation example "Effective thermal conductivity of a porous solid".

Builds the unit cell of a simple cubic array of spherical, fluid-filled pores with the CAD objects of
OneSimulation: a cube of 1 mm, written to porous_effective_conductivity_models.step with its faces
x = 0 and x = 1 mm as the groups hot and cold, and for the porosities 10, 20, 30 and 45 % the pore, a
sphere at the centre of the cube, written to porous_effective_conductivity_models_pore_<percent>.step.
"""

import math

from onesim.cad.d3 import Box, Sphere

side = 1e-3
cell = Box(length=side, width=side, height=side)
cell.grouped(hot=cell.boundaries.x_min, cold=cell.boundaries.x_max).save(
    "porous_effective_conductivity_models.step"
)
for percent in (10, 20, 30, 45):
    Sphere(
        radius=side * (3 * percent / 100 / (4 * math.pi)) ** (1 / 3),
        origin=(side / 2, side / 2, side / 2),
    ).save(f"porous_effective_conductivity_models_pore_{percent}.step")
