"""Geometry of the OneSimulation example "Cracking of a notched concrete beam in three-point bending".

Builds the beam (length 440 mm, depth W = 100 mm, thickness 50 mm) with a sharp notch of depth
50 mm cut in from the bottom at mid-span, 2 mm wide at the bottom face, and three bearing plates
10 mm wide and 5 mm high: two under the supports at a span of 400 mm (left_plate, right_plate)
and one on top at mid-span (load_plate). The part is written to notched_beam_phase_field.step; the
faces keep the names of their blocks (left_plate.y_min, right_plate.y_min, load_plate.y_max, ...),
the two flanks of the notch are the face notch.side.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

length, depth, thickness, span, notch, mouth, plate = (
    0.440,
    0.100,
    0.050,
    0.400,
    0.050,
    0.002,
    0.010,
)
middle = length / 2
pad = lambda x, y: Box(
    length=plate, width=0.005, height=thickness, origin=(x - plate / 2, y, 0.0)
)
wedge = Prism(
    profile=Profile.polygon(
        [(middle - mouth / 2, -0.004), (middle, notch), (middle + mouth / 2, -0.004)]
    ),
    height=thickness,
)
beam = (
    Box(length=length, width=depth, height=thickness)
    | pad(middle - span / 2, -0.005).named("left_plate")
    | pad(middle + span / 2, -0.005).named("right_plate")
    | pad(middle, depth).named("load_plate")
) - wedge.named("notch")
beam.save("notched_beam_phase_field.step")
