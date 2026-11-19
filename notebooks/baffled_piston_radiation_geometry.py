"""Geometry of the OneSimulation example "Baffled piston radiation".

Builds the air in front of a rigid piston (radius 50 mm) set in an infinite baffle, a hemisphere of
radius 0.6 m, as half a solid of revolution about the z axis with the CAD objects of OneSimulation
and writes baffled_piston_radiation.step with the faces piston, baffle and outer.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

piston, outer = 0.05, 0.6
outline = Path(
    (0.0, 0.0),
    (
        Segment((piston, 0.0), name="piston"),
        Segment((outer, 0.0), name="baffle"),
        Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
        Segment((0.0, 0.0), name="axis"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "baffled_piston_radiation.step"
)
