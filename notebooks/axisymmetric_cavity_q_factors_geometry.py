"""Geometry of the OneSimulation example "Q factors of axisymmetric cavity resonators".

Builds, as half solids of revolution about the z axis with the CAD objects of OneSimulation, the air
inside a cylindrical cavity of radius 25 mm and length 40 mm
(axisymmetric_cavity_q_factors_cylindrical.step) and inside a spherical cavity of radius 25 mm
(axisymmetric_cavity_q_factors_spherical.step). Both files name the metal surface `walls`.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius, length = 0.025, 0.040
outlines = {
    "cylindrical": Path(
        (0.0, 0.0),
        (
            Segment((radius, 0.0), name="walls"),
            Segment((radius, length), name="walls"),
            Segment((0.0, length), name="walls"),
            Segment((0.0, 0.0), name="axis"),
        ),
    ),
    "spherical": Path(
        (0.0, -radius),
        (
            Segment((0.0, radius), center=(0.0, 0.0), name="walls"),
            Segment((0.0, -radius), name="axis"),
        ),
    ),
}
for name, outline in outlines.items():
    Revolution(profile=Profile(outline), angle=np.pi).rotated(
        (0, 0, 1), -np.pi / 2
    ).save(f"axisymmetric_cavity_q_factors_{name}.step")
