"""Geometry of the OneSimulation example "Eigenmodes of a room".

Builds the air of two rooms with the CAD objects of OneSimulation: a living room of 5 m x 4 m
whose ceiling slopes from 2.4 m to 3.4 m (room_eigenmodes.step, faces floor, ceiling, wall and the gable walls bottom and top),
and a plain box of 5 m x 4 m x 2.8 m used to check the solver against the exact modes
(room_eigenmodes_box.step).
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Box, Prism

length, width, low, high = 5.0, 4.0, 2.4, 3.4
section = Profile(
    Path(
        (0.0, 0.0),
        (
            Segment((length, 0.0), name="floor"),
            Segment((length, high), name="wall"),
            Segment((0.0, low), name="ceiling"),
            Segment((0.0, 0.0), name="wall"),
        ),
    )
)
room = (
    Prism(profile=section, height=width)
    .rotated((1, 0, 0), np.pi / 2)
    .moved((0.0, width, 0.0))
)
room.save("room_eigenmodes.step")
Box(length=length, width=width, height=2.8).save("room_eigenmodes_box.step")
