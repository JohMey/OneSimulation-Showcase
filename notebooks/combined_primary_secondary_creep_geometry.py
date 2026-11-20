"""Geometry of the OneSimulation example "Primary and secondary creep in a hot thick-walled tube".

Builds a 90-degree sector of a thick-walled tube (inner radius 50 mm, outer radius 100 mm, a 20 mm
long ring of a long pipe, axis along z) with the CAD objects of OneSimulation, centred on the x axis,
and writes combined_primary_secondary_creep.step. The example computes on its meridian section y = 0. The
faces are named inner, outer, bottom (z = 0) and top (z = 20 mm).
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder

tube = AnnularCylinder(inner_radius=0.050, outer_radius=0.100, length=0.020)
sector = tube.sector(
    np.pi / 2, axis=(0.0, 0.0, 1.0), start=(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0)
)
sector = sector.grouped(
    inner=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius < 0.075
    ),
    outer=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius > 0.075
    ),
    bottom=sector.select(faces=lambda face: abs(face.center[2]) < 1e-9),
    top=sector.select(faces=lambda face: abs(face.center[2] - 0.020) < 1e-9),
)
sector.save("combined_primary_secondary_creep.step")
