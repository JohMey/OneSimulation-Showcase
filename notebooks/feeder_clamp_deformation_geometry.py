"""Geometry of the OneSimulation example "Deformation of a feeder clamp".

Builds the aluminium clamp (base plate with two bolt holes, upright web with fillets at its root
and at its neck, and an eye that carries the feeder) with the CAD objects of OneSimulation, names
the faces the simulation refers to and writes feeder_clamp_deformation.step.
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder, Box, UnionFromObjects

base_length, depth, plate, web, axis_height, outer, bore = (
    0.080,
    0.040,
    0.008,
    0.012,
    0.070,
    0.020,
    0.0125,
)

base = Box(
    length=base_length,
    width=depth,
    height=plate,
    origin=(-base_length / 2, -depth / 2, 0.0),
)
upright = Box(
    length=web,
    width=depth,
    height=axis_height - outer + 0.004 - plate,
    origin=(-web / 2, -depth / 2, plate),
)
eye = (
    AnnularCylinder(
        inner_radius=bore, outer_radius=outer, length=depth, origin=(0.0, 0.0, 0.0)
    )
    .rotated((1, 0, 0), -np.pi / 2)
    .moved((0.0, -depth / 2, axis_height))
)
clamp = UnionFromObjects(objects=[base, upright, eye])
clamp = clamp.fillet(
    0.005,
    edges=clamp.select(
        edges=lambda edge: abs(edge.start[2] - plate) < 1e-9
        and abs(edge.end[2] - plate) < 1e-9
        and abs(abs(edge.start[0]) - web / 2) < 1e-9
        and abs(edge.start[0] - edge.end[0]) < 1e-9
    ),
)
clamp = clamp.fillet(
    0.003,
    edges=clamp.select(
        edges=lambda edge: 0.045 < edge.start[2] < 0.06
        and abs(edge.start[2] - edge.end[2]) < 1e-9
        and abs(abs(edge.start[0]) - web / 2) < 1e-9
        and abs(edge.start[0] - edge.end[0]) < 1e-9
    ),
)
clamp = clamp.hole(0.009, at=(-0.028, 0.0, plate)).hole(0.009, at=(0.028, 0.0, plate))

faces = clamp.boundaries
clamp = clamp.grouped(
    bolts=faces["hole.outer_wall"] | faces["hole_2.outer_wall"],
    bore=clamp.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - bore) < 1e-9
    ),
    root=clamp.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - 0.005) < 1e-9
    ),
    neck=clamp.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - 0.003) < 1e-9
    ),
)
clamp.save("feeder_clamp_deformation.step")
