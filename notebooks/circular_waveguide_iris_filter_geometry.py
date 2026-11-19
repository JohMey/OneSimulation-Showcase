"""Geometry of the OneSimulation example "Circular waveguide iris filter".

Builds, as half solids of revolution about the z axis with the CAD objects of OneSimulation, the air
inside a circular waveguide of radius 20 mm with two annular irises (metal rings 2 mm thick with a
hole of 10 mm radius) that form a cavity 25 mm long, with 30 mm of guide on either side
(circular_waveguide_iris_filter.step; faces input and output are the ports, metal the wall and the
irises, axis the axis), and the closed cavity between the irises
(circular_waveguide_iris_filter_cavity.step, faces metal and axis).
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius, hole, iris, cavity, lead = 0.02, 0.01, 0.002, 0.025, 0.03
first, second = lead, lead + iris + cavity
end = second + iris + lead
outline = Path(
    (0.0, 0.0),
    (
        Segment((radius, 0.0), name="input"),
        Segment((radius, first), name="metal"),
        Segment((hole, first), name="metal"),
        Segment((hole, first + iris), name="metal"),
        Segment((radius, first + iris), name="metal"),
        Segment((radius, second), name="metal"),
        Segment((hole, second), name="metal"),
        Segment((hole, second + iris), name="metal"),
        Segment((radius, second + iris), name="metal"),
        Segment((radius, end), name="metal"),
        Segment((0.0, end), name="output"),
        Segment((0.0, 0.0), name="axis"),
    ),
)
closed = Path(
    (0.0, first + iris),
    (
        Segment((radius, first + iris), name="metal"),
        Segment((radius, second), name="metal"),
        Segment((0.0, second), name="metal"),
        Segment((0.0, first + iris), name="axis"),
    ),
)
for name, path in (("", outline), ("_cavity", closed)):
    Revolution(profile=Profile(path), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
        f"circular_waveguide_iris_filter{name}.step"
    )
