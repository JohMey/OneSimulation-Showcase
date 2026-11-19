"""Geometry of the OneSimulation example "Permeability of a hexagonal array of fibres".

Builds the periodic unit cell of a hexagonal array of parallel cylindrical fibres with the CAD
objects of OneSimulation: a rectangular prism 1 mm by 1.732 mm with a fibre along its axis and a
quarter fibre on each long edge, 0.1 mm long along the fibres. One file per solid fraction:
permeability_hexagonal_fibre_array.step (solid fraction 0.4) and
permeability_hexagonal_fibre_array_<percent>.step for 10 to 70 percent. The cell faces keep their
names x_min, x_max, y_min, y_max, z_min and z_max; the fibre surfaces are fibre.outer_wall.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

pitch, length = 1e-3, 1e-4
height = np.sqrt(3) * pitch
for percent in (10, 20, 30, 40, 50, 60, 70):
    radius = pitch * np.sqrt(percent / 100 * np.sqrt(3) / (2 * np.pi))
    cell = Box(length=pitch, width=height, height=length)
    for x, y in (
        (pitch / 2, height / 2),
        (0, 0),
        (pitch, 0),
        (0, height),
        (pitch, height),
    ):
        cell = cell - Cylinder(
            radius=radius, length=3 * length, origin=(x, y, -length)
        ).named("fibre")
    cell.save(f"permeability_hexagonal_fibre_array_{percent}.step")
    if percent == 40:
        cell.save("permeability_hexagonal_fibre_array.step")
