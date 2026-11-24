"""Geometry of the OneSimulation example "Biconical antenna".

Builds with the CAD objects of OneSimulation, as a half solid of revolution about the z axis, the air
around a biconical antenna: two metal cones of half angle 47 degrees and slant length 100 mm with flat
tops, tip to tip across a feed gap of 2 mm bridged by a thin feed wire (radius 1 mm), in a box reaching
450 mm from the axis and from the feed. File biconical_antenna_impedance.step names the faces port
(the surface of the feed wire, where the generator acts), metal (the cones) and outer (the walls of
the box); the axis is left unnamed.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

half_angle, slant, wire, gap, box = np.radians(47.0), 0.1, 0.001, 0.002, 0.45
top = (wire + slant * np.sin(half_angle), gap / 2 + slant * np.cos(half_angle))
outline = Path(
    (0.0, box),
    (
        Segment((0.0, top[1]), name="axis"),
        Segment(top, name="metal"),
        Segment((wire, gap / 2), name="metal"),
        Segment((wire, -gap / 2), name="port"),
        Segment((top[0], -top[1]), name="metal"),
        Segment((0.0, -top[1]), name="metal"),
        Segment((0.0, -box), name="axis"),
        Segment((box, -box), name="outer"),
        Segment((box, box), name="outer"),
        Segment((0.0, box), name="outer"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "biconical_antenna_impedance.step"
)
