"""Geometry of the OneSimulation example "Modifying an imported CAD part by deformation".

Builds a clamp block (100 x 40 x 30 mm with a bore of 16 mm across it) with the CAD objects of
OneSimulation, names the faces the example moves or holds and writes deforming_imported_cad_part.step.
"""

from onesim.cad.d3 import Box, Cylinder

block = Box(length=0.1, width=0.04, height=0.03) - Cylinder(
    radius=0.008, length=0.04, origin=(0.03, 0.0, 0.015), axis=(0, 1, 0)
).named("bore")
block = block.grouped(
    held=block.boundaries.x_min,
    moved=block.boundaries.x_max,
    bore=block.boundaries["bore.outer_wall"],
    top=block.boundaries.z_max,
)
block.save("deforming_imported_cad_part.step")
