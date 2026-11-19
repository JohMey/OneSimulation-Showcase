"""Geometry of the OneSimulation example "Axisymmetric steady conduction in a thick ring".

Builds half of a thick-walled steel ring (bore radius 20 mm, outer radius 100 mm, height 140 mm)
as a solid of revolution about the z axis with the CAD objects of OneSimulation and writes
axisymmetric_ring_steady_conduction.step. The bore is split into the heated band (z from 40 to
100 mm) and the insulated parts below and above it; the cooled faces are bottom, outer and top.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

bore, outer, height, band = 0.02, 0.1, 0.14, (0.04, 0.10)
outline = Path(
    (bore, 0.0),
    (
        Segment((outer, 0.0), name="bottom"),
        Segment((outer, height), name="outer"),
        Segment((bore, height), name="top"),
        Segment((bore, band[1]), name="bore_above"),
        Segment((bore, band[0]), name="heated"),
        Segment((bore, 0.0), name="bore_below"),
    ),
)
ring = Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2)
ring.save("axisymmetric_ring_steady_conduction.step")
