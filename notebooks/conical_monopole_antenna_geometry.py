"""Geometry of the OneSimulation example "Monoconical antenna on a finite ground plane".

Builds with the CAD objects of OneSimulation, as a half solid of revolution about the z axis, the air
around a monoconical antenna: a metal cone of half angle 47 degrees and slant length 100 mm with a
flat top, standing on a thin feed wire (radius 1 mm, gap 2 mm) above a ground disc of radius 150 mm
and thickness 2 mm, in a box reaching 450 mm from the axis and from the ground plane.
File conical_monopole_antenna.step names the faces port (the surface of the feed wire, where the
generator acts), metal (cone and ground) and outer (the walls of the box); the axis is left unnamed.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

half_angle, slant, wire, gap, ground, plate, box = (
    np.radians(47.0),
    0.1,
    0.001,
    0.002,
    0.15,
    0.002,
    0.45,
)
top = (wire + slant * np.sin(half_angle), gap + slant * np.cos(half_angle))
outline = Path(
    (0.0, box),
    (
        Segment((0.0, top[1]), name="axis"),
        Segment(top, name="metal"),
        Segment((wire, gap), name="metal"),
        Segment((wire, 0.0), name="port"),
        Segment((ground, 0.0), name="metal"),
        Segment((ground, -plate), name="metal"),
        Segment((0.0, -plate), name="metal"),
        Segment((0.0, -box), name="axis"),
        Segment((box, -box), name="outer"),
        Segment((box, box), name="outer"),
        Segment((0.0, box), name="outer"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "conical_monopole_antenna.step"
)
