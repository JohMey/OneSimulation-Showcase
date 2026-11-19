"""Geometry of the OneSimulation example "Homogenized permeability of a simple cubic sphere packing".

Builds a quarter of the periodic unit cell of a simple cubic array of spheres with the CAD objects
of OneSimulation: the cell has the side 1 mm, the quarter holds the half 1 mm x 0.5 mm x 0.5 mm with
a quarter sphere centred on its edge at (0.5, 0, 0) mm. One file per solid fraction:
periodic_sphere_array_permeability.step (solid fraction 0.216) and
periodic_sphere_array_permeability_<per mille>.step for 64 to 450 per mille. The faces keep their
names x_min, x_max, y_min, y_max, z_min and z_max, the sphere surface is sphere.surface.
"""

import numpy as np
from onesim.cad.d3 import Box, Sphere

side = 1e-3
for per_mille in (64, 125, 216, 343, 450):
    radius = side * (3 * per_mille / 1000 / (4 * np.pi)) ** (1 / 3)
    quarter = Box(length=side, width=side / 2, height=side / 2) - Sphere(
        radius=radius, origin=(side / 2, 0.0, 0.0)
    ).named("sphere")
    quarter.save(f"periodic_sphere_array_permeability_{per_mille}.step")
    if per_mille == 216:
        quarter.save("periodic_sphere_array_permeability.step")
