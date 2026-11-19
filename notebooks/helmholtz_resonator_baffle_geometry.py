"""Geometry of the OneSimulation example "Helmholtz resonator in a wall".

Builds the air of a Helmholtz resonator set into a rigid wall - a cylindrical cavity (radius 60 mm,
depth 60 mm) behind a neck (radius 20 mm, length 40 mm) - together with a hemisphere of air of
radius 0.4 m in front of the wall, as half a solid of revolution about the z axis with the CAD
objects of OneSimulation. It writes helmholtz_resonator_baffle.step with the faces cavity, neck,
wall and outer.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

cavity_radius, cavity_depth, neck_radius, neck_length, outer = (
    0.06,
    0.06,
    0.02,
    0.04,
    0.4,
)
bottom = -neck_length - cavity_depth
outline = Path(
    (0.0, bottom),
    (
        Segment((cavity_radius, bottom), name="cavity"),
        Segment((cavity_radius, -neck_length), name="cavity"),
        Segment((neck_radius, -neck_length), name="cavity"),
        Segment((neck_radius, 0.0), name="neck"),
        Segment((outer, 0.0), name="wall"),
        Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
        Segment((0.0, bottom), name="axis"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "helmholtz_resonator_baffle.step"
)
