"""Input file of the OneSimulation example "Combining CAD geometry with meshes".

Writes cad_and_mesh_combined.stl: the triangulated surface of a knob, an ellipsoid with half axes
20, 20 and 12 mm, as a scanner or another program would deliver it (the zero surface of a level
set sampled on a grid of 2 mm, built with the CAD objects of OneSimulation).
"""

import numpy as np
from onesim.cad.d3 import isosurface

grid = np.linspace(-0.022, 0.022, 23)
x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
knob = isosurface(
    1.0 - np.sqrt((x / 0.02) ** 2 + (y / 0.02) ** 2 + (z / 0.012) ** 2),
    grid[1] - grid[0],
    level=0.0,
    origin=(grid[0], grid[0], grid[0]),
)
knob.save("cad_and_mesh_combined.stl")
