"""Geometry of the OneSimulation example "Axisymmetric transient heating of a cylinder".

Builds half of a solid steel cylinder, 0.3 m in radius and 0.4 m high, as a solid of revolution
about the z axis with the CAD objects of OneSimulation and writes
axisymmetric_cylinder_transient_heating.step. The heated surface consists of the faces bottom,
outer and top; the meridian plane y = 0 of the part is the computational domain.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius, height = 0.3, 0.4
outline = Path(
    (0.0, 0.0),
    (
        Segment((radius, 0.0), name="bottom"),
        Segment((radius, height), name="outer"),
        Segment((0.0, height), name="top"),
        Segment((0.0, 0.0), name="axis"),
    ),
)
cylinder = Revolution(profile=Profile(outline), angle=np.pi).rotated(
    (0, 0, 1), -np.pi / 2
)
cylinder.save("axisymmetric_cylinder_transient_heating.step")
