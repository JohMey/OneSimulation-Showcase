"""Geometry of the OneSimulation example "Stress relaxation of a viscoelastic tube under a temperature rise".

Builds a 90-degree sector of a thick-walled tube (inner radius 20 mm, outer radius 40 mm, a 10 mm
long ring of a long tube, axis along z) with the CAD objects of OneSimulation, centred on the x axis,
and writes viscoelastic_tube_thermal_relaxation.step. The example computes on its meridian section
y = 0. The faces are named inner, outer, bottom and top.
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder

tube = AnnularCylinder(inner_radius=0.020, outer_radius=0.040, length=0.010)
sector = tube.sector(
    np.pi / 2, axis=(0.0, 0.0, 1.0), start=(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0)
)
sector = sector.grouped(
    inner=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius < 0.03
    ),
    outer=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius > 0.03
    ),
    bottom=sector.select(faces=lambda face: abs(face.center[2]) < 1e-9),
    top=sector.select(faces=lambda face: abs(face.center[2] - 0.010) < 1e-9),
)
sector.save("viscoelastic_tube_thermal_relaxation.step")
