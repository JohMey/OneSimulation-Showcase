"""Geometry of the OneSimulation example "Homogenized thermal properties of a unidirectional fibre composite".

Builds the unit cell of a square array of parallel fibres with the CAD objects of OneSimulation: a
square prism of 10 µm x 10 µm x 5 µm, written to fiber_composite_thermal_homogenization.step, and the
fibre of 3.99 µm radius along its axis, which fills half of the cell, written to
fiber_composite_thermal_homogenization_fibre.step. The cell faces x = 0 and x = 10 µm are the groups
left and right, the faces y = 0 and y = 10 µm the group sides, the faces z = 0 and z = 5 µm the
groups bottom and top.
"""

import math

from onesim.cad.d3 import Box, Cylinder

side, depth, fraction = 10e-6, 5e-6, 0.5
cell = Box(length=side, width=side, height=depth)
faces = cell.boundaries
cell = cell.grouped(
    left=faces.x_min,
    right=faces.x_max,
    sides=faces.y_min | faces.y_max,
    bottom=faces.z_min,
    top=faces.z_max,
)
cell.save("fiber_composite_thermal_homogenization.step")
Cylinder(
    radius=side * math.sqrt(fraction / math.pi),
    length=depth,
    origin=(side / 2, side / 2, 0.0),
).save("fiber_composite_thermal_homogenization_fibre.step")
