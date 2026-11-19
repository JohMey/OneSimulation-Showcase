"""Geometry of the OneSimulation example "Permeability of a square array of fibres".

Builds the periodic unit cell of a square array of parallel cylindrical fibres with the CAD objects
of OneSimulation: a square prism of side 1 mm around one fibre, 0.1 mm long along the fibre. One
file per solid fraction: permeability_square_fibre_array.step (solid fraction 0.3) and
permeability_square_fibre_array_<percent>.step for 5 to 60 percent. The cell faces keep their names
x_min, x_max, y_min, y_max, z_min and z_max, the fibre surface is fibre.outer_wall.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

side, thickness = 1e-3, 1e-4
for percent in (5, 10, 20, 30, 40, 50, 60):
    radius = side * np.sqrt(percent / 100 / np.pi)
    cell = Box(length=side, width=side, height=thickness) - Cylinder(
        radius=radius, length=3 * thickness, origin=(side / 2, side / 2, -thickness)
    ).named("fibre")
    cell.save(f"permeability_square_fibre_array_{percent}.step")
    if percent == 30:
        cell.save("permeability_square_fibre_array.step")
