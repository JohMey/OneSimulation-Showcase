"""Geometry of the OneSimulation example "Stress intensity factor of a single-edge crack".

Builds a plate of width W = 100 mm, height 400 mm and thickness 10 mm with an edge crack at half
height, once for each crack length a/W = 0.2, 0.3, 0.4 and 0.5, with the CAD objects of
OneSimulation. The crack is a sharp wedge cut in from the left side, 0.2 mm wide at its mouth.
Each part is written to edge_crack_stress_intensity_<a/W in percent>.step; its faces keep the
names x_min, x_max, y_min and y_max, the two crack flanks are the face crack.side.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

width, height, thickness, mouth = 0.100, 0.400, 0.010, 0.0002
for ratio in (0.2, 0.3, 0.4, 0.5):
    tip = ratio * width
    wedge = Prism(
        profile=Profile.polygon(
            [
                (-0.1 * width, (height - mouth) / 2),
                (tip, height / 2),
                (-0.1 * width, (height + mouth) / 2),
            ]
        ),
        height=thickness,
    )
    plate = Box(length=width, width=height, height=thickness) - wedge.named("crack")
    plate.save(f"edge_crack_stress_intensity_{round(100 * ratio)}.step")
