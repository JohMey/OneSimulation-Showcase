"""Geometry of the OneSimulation example "Thick elliptic plate under pressure (NAFEMS LE10)".

Builds the quarter of the elliptic ring plate (hole 2 m x 1 m, outer edge 3.25 m x 2.75 m half axes,
0.6 m thick, centred on z = 0) with the CAD objects of OneSimulation and writes
thick_plate_pressure_le10.step. The file names the faces top, bottom, hole, outer, symmetry_x and
symmetry_y.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

hole, outer_edge, thickness = (2.0, 1.0), (3.25, 2.75), 0.6
ring = Prism(
    profile=Profile.ellipse(*outer_edge, name="outer").with_holes(
        Profile.ellipse(*hole, name="hole")
    ),
    height=thickness,
    origin=(0.0, 0.0, -thickness / 2),
)
plate = ring & Box(length=4.0, width=4.0, height=1.0, origin=(0.0, 0.0, -0.5))
plate = plate.grouped(
    symmetry_x=plate.boundaries["part.x_min"], symmetry_y=plate.boundaries["part.y_min"]
)
plate.save("thick_plate_pressure_le10.step")
