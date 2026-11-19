"""Geometry of the OneSimulation example "Friction stir welding of an aluminium plate".

Builds the aluminium plate, 300 mm long in the welding direction (x), 200 mm wide and 6 mm thick,
with the CAD objects of OneSimulation and writes friction_stir_welding_plate.step. The tool axis
is the z axis; the plate reaches 100 mm behind and 200 mm ahead of it. The four cut edges form the
group edges and the two large faces the group faces.
"""

from onesim.cad.d3 import Box

plate = Box(length=0.3, width=0.2, height=0.006, origin=(-0.1, -0.1, 0.0))
f = plate.boundaries
plate = plate.grouped(
    edges=f.x_min | f.x_max | f.y_min | f.y_max, faces=f.z_min | f.z_max
)
plate.save("friction_stir_welding_plate.step")
