"""Geometry of the OneSimulation example "Elliptic membrane under outward pressure (NAFEMS LE1)".

Builds the quarter of the elliptic membrane (inner edge the ellipse with half axes 2 m x 1 m,
outer edge 3.25 m x 2.75 m, 0.1 m thick) with the CAD objects of OneSimulation and writes
elliptic_membrane_le1.step. The faces are named inner, outer, symmetry_x and symmetry_y.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

inner, outer, thickness = (2.0, 1.0), (3.25, 2.75), 0.1
ring = Prism(
    profile=Profile.ellipse(*outer, name="outer").with_holes(
        Profile.ellipse(*inner, name="inner")
    ),
    height=thickness,
)
membrane = ring & Box(length=4.0, width=4.0, height=thickness)
membrane = membrane.grouped(
    symmetry_x=membrane.boundaries["part.x_min"],
    symmetry_y=membrane.boundaries["part.y_min"],
)
membrane.save("elliptic_membrane_le1.step")
