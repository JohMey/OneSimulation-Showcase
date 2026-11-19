"""Geometry of the OneSimulation example "Mandel-Cryer effect in a consolidating sphere".

Builds the upper half of one half of the soil sphere (radius 1 m: x >= 0, z >= 0) as a solid of
revolution with the CAD objects of OneSimulation and writes mandel_cryer_consolidation.step. The
quarter disc that is revolved names the faces: surface (the draining sphere), equator (the plane
z = 0) and axis (the flat face through the axis of symmetry z). The example computes on the
meridian plane y = 0, with x the radius and z the height.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius = 1.0
outline = Path(
    (0.0, 0.0),
    (
        Segment((radius, 0.0), name="equator"),
        Segment((0.0, radius), center=(0.0, 0.0), name="surface"),
        Segment((0.0, 0.0), name="axis"),
    ),
)
quarter = Revolution(profile=Profile(outline), angle=np.pi).rotated(
    (0, 0, 1), -np.pi / 2
)
quarter.save("mandel_cryer_consolidation.step")
