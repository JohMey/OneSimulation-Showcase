"""Geometry of the OneSimulation example "Whispering-gallery modes of a dielectric microsphere".

Builds, as half solids of revolution about the z axis with the CAD objects of OneSimulation, a ball
of air of radius 7 um (whispering_gallery_microsphere.step, outer surface `outer`) with a silica
microsphere of radius 4 um inside it (whispering_gallery_microsphere_sphere.step). The outer 1.2 um
of the air ball becomes the absorbing layer in the notebook.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

um = 1e-6
sphere, air = 4 * um, 7 * um
half = lambda radius, name: Revolution(
    profile=Profile(
        Path(
            (0.0, -radius),
            (
                Segment((0.0, radius), center=(0.0, 0.0), name=name),
                Segment((0.0, -radius), name="axis"),
            ),
        )
    ),
    angle=np.pi,
).rotated((0, 0, 1), -np.pi / 2)
half(air, "outer").save("whispering_gallery_microsphere.step")
half(sphere, "surface").save("whispering_gallery_microsphere_sphere.step")
