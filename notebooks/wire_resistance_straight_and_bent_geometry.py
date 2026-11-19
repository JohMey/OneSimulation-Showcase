"""Geometry of the OneSimulation example "Resistance of a straight and a bent copper wire".

Builds two pieces of a copper wire of 1 mm radius with the CAD objects of OneSimulation: a straight
piece 20 mm long, written to wire_resistance_straight_and_bent.step with the faces bottom, top and
outer_wall, and a quarter bend with a centreline radius of 2 mm, written to
wire_resistance_straight_and_bent_bend.step with the end faces start and end.
"""

from onesim.cad.d3 import Box, Cylinder, Torus

radius, length, bend = 1.0e-3, 20.0e-3, 2.0e-3
Cylinder(radius=radius, length=length).save("wire_resistance_straight_and_bent.step")
quarter = Torus(major_radius=bend, minor_radius=radius) & Box(
    length=2 * bend, width=2 * bend, height=2 * radius, origin=(0.0, 0.0, -radius)
).named("cut")
faces = quarter.boundaries
quarter = quarter.grouped(start=faces["cut.y_min"], end=faces["cut.x_min"])
quarter.save("wire_resistance_straight_and_bent_bend.step")
