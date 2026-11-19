"""Geometry of the OneSimulation example "Radiation impedance of a flanged pipe".

Builds the air of a pipe (radius 20 mm, length 250 mm) that opens through an infinite flange into a
hemisphere of radius 300 mm, and the air of the pipe alone, as half solids of revolution about the z
axis with the CAD objects of OneSimulation. The first is written to
flanged_pipe_radiation_impedance.step with the faces piston, wall, flange and outer; the second to
flanged_pipe_radiation_impedance_pipe.step with the faces piston, wall and opening.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius, length, outer = 0.02, 0.25, 0.30
outlines = {
    "flanged_pipe_radiation_impedance": Path(
        (0.0, 0.0),
        (
            Segment((radius, 0.0), name="piston"),
            Segment((radius, length), name="wall"),
            Segment((outer, length), name="flange"),
            Segment((0.0, length + outer), center=(0.0, length), name="outer"),
            Segment((0.0, 0.0), name="axis"),
        ),
    ),
    "flanged_pipe_radiation_impedance_pipe": Path(
        (0.0, 0.0),
        (
            Segment((radius, 0.0), name="piston"),
            Segment((radius, length), name="wall"),
            Segment((0.0, length), name="opening"),
            Segment((0.0, 0.0), name="axis"),
        ),
    ),
}
for name, outline in outlines.items():
    Revolution(profile=Profile(outline), angle=np.pi).rotated(
        (0, 0, 1), -np.pi / 2
    ).save(f"{name}.step")
