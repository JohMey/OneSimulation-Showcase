"""Geometry of the OneSimulation example "Magnetic field of a long wire in a square air domain".

Builds a 2.5 mm slab of the cross-section with the CAD objects of OneSimulation: the square air
region of side 20 cm around the axis, written to wire_field_square_domain.step, and the round wire
of radius 1 cm, written to wire_field_square_domain_wire.step. The four walls of the air region
form the group walls.
"""

from onesim.cad.d3 import Box, Cylinder

side, wire_radius, thickness = 0.20, 0.01, 0.0025
air = Box(length=side, width=side, height=thickness, origin=(-side / 2, -side / 2, 0.0))
faces = air.boundaries
air = air.grouped(walls=faces.x_min | faces.x_max | faces.y_min | faces.y_max)
air.save("wire_field_square_domain.step")
Cylinder(radius=wire_radius, length=thickness).save("wire_field_square_domain_wire.step")
