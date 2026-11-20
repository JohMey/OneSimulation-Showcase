"""Geometry of the OneSimulation example "Isotropic compression of clay with Modified Cam-Clay".

Builds a 90-degree sector of a cylindrical soil sample (38 mm diameter, 76 mm high, axis along z)
with the CAD objects of OneSimulation, centred on the x axis, and writes
isotropic_compression_cam_clay.step. The example computes on its meridian section y = 0. The faces
are named mantle, bottom (z = 0) and top (z = 76 mm).
"""

import numpy as np
from onesim.cad.d3 import Cylinder

sample = Cylinder(radius=0.019, length=0.076)
sector = sample.sector(
    np.pi / 2, axis=(0.0, 0.0, 1.0), start=(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0)
)
sector = sector.grouped(
    mantle=sector.select(faces=lambda face: face.kind == "cylinder"),
    bottom=sector.select(
        faces=lambda face: abs(face.center[2]) < 1e-9 and face.kind == "plane"
    ),
    top=sector.select(
        faces=lambda face: abs(face.center[2] - 0.076) < 1e-9 and face.kind == "plane"
    ),
)
sector.save("isotropic_compression_cam_clay.step")
