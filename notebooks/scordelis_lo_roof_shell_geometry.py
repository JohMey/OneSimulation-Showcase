"""Geometry of the OneSimulation example "Scordelis-Lo roof".

Builds the cylindrical roof of the benchmark, radius 25 m to the mid-surface, 50 m long along x,
spanning 40 degrees on either side of the crown and 0.25 m thick, as a thin solid with the CAD
objects of OneSimulation, and writes scordelis_lo_roof_shell.step. The faces are named diaphragm
(the two curved ends at x = -25 m and x = 25 m) and free_edge (the two straight edges).
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder

radius, length, thickness, half_angle = 25.0, 50.0, 0.25, np.radians(40.0)
tube = AnnularCylinder(
    inner_radius=radius - thickness / 2,
    outer_radius=radius + thickness / 2,
    length=length,
    origin=(-length / 2, 0.0, 0.0),
    axis=(1.0, 0.0, 0.0),
)
roof = tube.sector(
    2 * half_angle,
    axis=(1.0, 0.0, 0.0),
    start=(0.0, np.sin(half_angle), np.cos(half_angle)),
)
roof = roof.grouped(
    diaphragm=roof.select(
        faces=lambda face: face.kind == "plane"
        and abs(abs(face.center[0]) - length / 2) < 1e-6
    ),
    free_edge=roof.boundaries.sector_start | roof.boundaries.sector_end,
)
roof.save("scordelis_lo_roof_shell.step")
