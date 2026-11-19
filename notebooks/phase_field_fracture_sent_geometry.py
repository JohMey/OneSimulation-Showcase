"""Geometry of the OneSimulation example "Phase-field fracture of a single-edge notched specimen".

Builds the square specimen (side 1 mm, thickness 1 mm) with the CAD objects of OneSimulation. The
notch is a sharp wedge cut in from the left side to the centre, 4.8 micrometres wide at its mouth.
The part is written to phase_field_fracture_sent.step; its faces keep the names x_min, x_max,
y_min and y_max, the two flanks of the notch are the face notch.side.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

side, thickness, mouth = 1.0e-3, 1.0e-3, 4.8e-6
wedge = Prism(
    profile=Profile.polygon(
        [
            (-0.1 * side, (side - mouth) / 2),
            (side / 2, side / 2),
            (-0.1 * side, (side + mouth) / 2),
        ]
    ),
    height=thickness,
)
specimen = Box(length=side, width=side, height=thickness) - wedge.named("notch")
specimen.save("phase_field_fracture_sent.step")
