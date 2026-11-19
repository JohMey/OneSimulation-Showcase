"""Geometry of the OneSimulation example "Monopole antenna fed by a magnetic frill".

Builds, as a half solid of revolution about the z axis with the CAD objects of OneSimulation, the
air around a monopole 70 mm high and 1.5 mm in radius on an infinite ground plane, out to a
hemisphere of 0.3 m, together with the 10 mm of coaxial line (outer radius 3.45 mm, 50 ohm) that
feeds it through the ground plane (monopole_magnetic_frill_feed.step; faces antenna (wire and inner
conductor), ground (ground plane and outer conductor), feed (the coaxial port), outer (the
hemisphere) and axis).
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

height, wire, shield, line, outer = 0.07, 0.0015, 0.00345, 0.01, 0.3
outline = Path(
    (0.0, height),
    (
        Segment((wire, height), name="antenna"),
        Segment((wire, -line), name="antenna"),
        Segment((shield, -line), name="feed"),
        Segment((shield, 0.0), name="ground"),
        Segment((outer, 0.0), name="ground"),
        Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
        Segment((0.0, height), name="axis"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "monopole_magnetic_frill_feed.step"
)
