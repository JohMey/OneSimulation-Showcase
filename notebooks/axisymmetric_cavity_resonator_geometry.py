"""Geometry of the OneSimulation example "Axisymmetric cavity resonator".

Builds the air inside a closed cylindrical cavity (pillbox) of radius 50 mm and height 80 mm as half
a solid of revolution about the z axis with the CAD objects of OneSimulation, and writes
axisymmetric_cavity_resonator.step. Its meridian plane is a rectangle; the metal faces (the side
wall and the two end plates) are named `walls`.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius, height = 0.05, 0.08
outline = Path(
    (0.0, 0.0),
    (
        Segment((radius, 0.0), name="walls"),
        Segment((radius, height), name="walls"),
        Segment((0.0, height), name="walls"),
        Segment((0.0, 0.0), name="axis"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "axisymmetric_cavity_resonator.step"
)
