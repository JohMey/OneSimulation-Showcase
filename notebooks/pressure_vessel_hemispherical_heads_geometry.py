"""Geometry of the OneSimulation example "Pressure vessel with hemispherical heads".

Builds one octant of the vessel wall (inner radius 0.5 m, wall 25 mm, half length 1 m, hemispherical
head) as a solid of revolution with the CAD objects of OneSimulation and writes
pressure_vessel_hemispherical_heads.step. The file names the faces inside, outside, symmetry_x,
symmetry_y and symmetry_z.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

inner, wall, length = 0.5, 0.025, 1.0
outer = inner + wall
wall_section = Profile(
    Path(
        (inner, 0.0),
        (
            Segment((outer, 0.0), name="symmetry_z"),
            Segment((outer, length), name="outside"),
            Segment((0.0, length + outer), center=(0.0, length), name="outside"),
            Segment((0.0, length + inner), name="axis"),
            Segment(
                (inner, length),
                center=(0.0, length),
                counterclockwise=False,
                name="inside",
            ),
            Segment((inner, 0.0), name="inside"),
        ),
    )
)
vessel = Revolution(profile=wall_section, angle=np.pi / 2)
vessel = vessel.grouped(
    symmetry_y=vessel.boundaries.start, symmetry_x=vessel.boundaries.end
)
vessel.save("pressure_vessel_hemispherical_heads.step")
