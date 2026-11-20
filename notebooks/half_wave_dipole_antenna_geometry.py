"""Geometry of the OneSimulation example "Half-wave dipole antenna".

Builds with the CAD objects of OneSimulation the air around a centre-fed dipole antenna as one half
of a solid of revolution about the z axis: a ball of radius 350 mm (the outer 100 mm become an
absorbing layer) without the antenna, a perfectly conducting cylinder of radius 1 mm and length
144 mm with a feed gap of 2 mm in its middle. Faces in half_wave_dipole_antenna.step: outer (the
sphere), wire (the surface of the two halves), gap (the band of the feed gap) and axis (the cut
along the z axis).
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

outer, length, radius, gap = 0.35, 0.144, 0.001, 0.002
outline = Path(
    (0.0, -outer),
    (
        Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
        Segment((0.0, length / 2), name="axis"),
        Segment((radius, length / 2), name="wire"),
        Segment((radius, gap / 2), name="wire"),
        Segment((radius, -gap / 2), name="gap"),
        Segment((radius, -length / 2), name="wire"),
        Segment((0.0, -length / 2), name="wire"),
        Segment((0.0, -outer), name="axis"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "half_wave_dipole_antenna.step"
)
