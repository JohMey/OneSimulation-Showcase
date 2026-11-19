"""Geometry of the OneSimulation example "Plane wave scattered by a rigid sphere".

Builds a quarter of the air around a rigid sphere of radius 0.1 m, out to an outer sphere, as a
solid of revolution about the z axis with the CAD objects of OneSimulation. The outer sphere has a
radius of 0.4 m in rigid_sphere_scattering.step and of 0.7 m in rigid_sphere_scattering_large.step;
both name the faces scatterer, outer, symmetry_xz and symmetry_yz.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius = 0.1
for name, outer in (
    ("rigid_sphere_scattering", 0.4),
    ("rigid_sphere_scattering_large", 0.7),
):
    outline = Path(
        (0.0, -outer),
        (
            Segment((outer, 0.0), center=(0.0, 0.0), name="outer"),
            Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
            Segment((0.0, radius), name="axis"),
            Segment(
                (radius, 0.0),
                center=(0.0, 0.0),
                counterclockwise=False,
                name="scatterer",
            ),
            Segment(
                (0.0, -radius),
                center=(0.0, 0.0),
                counterclockwise=False,
                name="scatterer",
            ),
            Segment((0.0, -outer), name="axis"),
        ),
    )
    air = Revolution(profile=Profile(outline), angle=np.pi / 2)
    air = air.grouped(symmetry_xz=air.boundaries.start, symmetry_yz=air.boundaries.end)
    air.save(f"{name}.step")
