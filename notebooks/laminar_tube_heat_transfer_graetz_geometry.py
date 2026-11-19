"""Geometry of the OneSimulation example "Heat transfer in laminar flow through a circular tube".

Builds one quarter of a circular tube of 2 mm diameter and 20 mm length with the CAD objects of
OneSimulation and writes laminar_tube_heat_transfer_graetz.step. The flow runs along z; the faces
form the groups inlet (z = 0), outlet (z = 20 mm), wall and symmetry (the two cut planes through
the axis).
"""

from onesim.cad.d3 import Box, Cylinder

radius, length = 1.0e-3, 20.0e-3
tube = Cylinder(radius=radius, length=length).named("tube") & Box(
    length=radius, width=radius, height=length
).named("cut")
faces = tube.boundaries
tube.grouped(
    inlet=tube.select(faces=lambda face: face.id in ("tube.bottom", "cut.z_min")),
    outlet=tube.select(faces=lambda face: face.id in ("tube.top", "cut.z_max")),
    wall=faces["tube.outer_wall"],
    symmetry=faces["cut.x_min"] | faces["cut.y_min"],
).save("laminar_tube_heat_transfer_graetz.step")
