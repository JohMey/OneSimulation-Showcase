"""Geometry of the OneSimulation example "Natural frequencies of a free thick cylinder (NAFEMS FV41)".

Builds a 90-degree sector of the hollow cylinder of the benchmark (inner radius 1.8 m, outer radius
2.2 m, length 10 m, axis along z) with the CAD objects of OneSimulation, centred on the x axis,
and writes free_cylinder_axisymmetric_modes.step. The example computes on its meridian section
y = 0. The faces are named inner, outer, bottom (z = 0) and top (z = 10 m).
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder

tube = AnnularCylinder(inner_radius=1.8, outer_radius=2.2, length=10.0)
sector = tube.sector(
    np.pi / 2, axis=(0.0, 0.0, 1.0), start=(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0)
)
sector = sector.grouped(
    inner=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius < 2.0
    ),
    outer=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius > 2.0
    ),
    bottom=sector.select(faces=lambda face: abs(face.center[2]) < 1e-9),
    top=sector.select(faces=lambda face: abs(face.center[2] - 10.0) < 1e-9),
)
sector.save("free_cylinder_axisymmetric_modes.step")
