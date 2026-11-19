"""Geometry of the OneSimulation example "Brittle fracture of a plate with a hole and a notch".

Builds a plate 65 mm wide, 120 mm high and 1 mm thick with an off-centre hole of 20 mm diameter
at (36.5 mm, 51 mm) and a sharp notch cut 10 mm in from the left edge at a height of 65 mm,
1 mm wide at the edge, with the CAD objects of OneSimulation. The part is written to
holed_plate_phase_field.step; the faces keep the names x_min, x_max, y_min, y_max, the hole wall
is hole.outer_wall and the notch flanks are notch.side.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Cylinder, Prism

mm = 1e-3
width, height, thickness = 65 * mm, 120 * mm, 1 * mm
hole = Cylinder(
    radius=10 * mm, length=4 * thickness, origin=(36.5 * mm, 51 * mm, -thickness)
).named("hole")
notch = Prism(
    profile=Profile.polygon(
        [(-5 * mm, 64.5 * mm), (10 * mm, 65 * mm), (-5 * mm, 65.5 * mm)]
    ),
    height=thickness,
).named("notch")
(Box(length=width, width=height, height=thickness) - hole - notch).save(
    "holed_plate_phase_field.step"
)
