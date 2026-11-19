"""Geometry of the OneSimulation example "Eigenmodes in a muffler chamber".

Builds the air of an oval muffler chamber, an elliptic cylinder with semi-axes 100 mm and 60 mm and
length 400 mm, with the CAD objects of OneSimulation (a circular cylinder scaled across) and writes
muffler_chamber_modes.step with the faces mantle and ends.
"""

from onesim.cad.d3 import Cylinder

chamber = Cylinder(radius=0.1, length=0.4).scaled((1.0, 0.6, 1.0))
faces = chamber.boundaries
chamber.grouped(mantle=faces.outer_wall, ends=faces.bottom | faces.top).save(
    "muffler_chamber_modes.step"
)
