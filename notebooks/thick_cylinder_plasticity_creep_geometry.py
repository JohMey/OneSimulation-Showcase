"""Geometry of the OneSimulation example "Plasticity and creep in a heated thick-walled cylinder".

Builds a 90-degree sector of a thick-walled cylinder (inner radius 100 mm, outer radius 200 mm, a
20 mm long slice of a long cylinder, axis along z) with the CAD objects of OneSimulation, names its
faces and writes thick_cylinder_plasticity_creep.step.
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder

tube = AnnularCylinder(inner_radius=0.100, outer_radius=0.200, length=0.020)
sector = tube.sector(np.pi / 2, axis=(0.0, 0.0, 1.0), start=(1.0, 0.0, 0.0))
sector = sector.grouped(
    inner=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius < 0.15
    ),
    outer=sector.select(
        faces=lambda face: face.kind == "cylinder" and face.radius > 0.15
    ),
    symmetry_x=sector.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0]) < 1e-9
    ),
    symmetry_y=sector.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-9
    ),
    ends=sector.select(
        faces=lambda face: face.kind == "plane"
        and min(abs(face.center[2]), abs(face.center[2] - 0.020)) < 1e-9
    ),
)
sector.save("thick_cylinder_plasticity_creep.step")
