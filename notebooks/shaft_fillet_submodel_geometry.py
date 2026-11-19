"""Geometry of the OneSimulation example "Submodel of a stepped shaft".

Builds a stepped steel shaft with the CAD objects of OneSimulation: 40 mm diameter over the first
120 mm, 60 mm diameter over the last 80 mm, joined by a shoulder fillet of 4 mm radius; the axis is
x. The whole shaft is written to shaft_fillet_submodel.step with the faces loaded (x = 0), clamped
(x = 200 mm) and fillet. The submodel, the part of the shaft between x = 100 mm and x = 140 mm
around the shoulder, is written to shaft_fillet_submodel_local.step with the faces cut (the two
cut planes) and fillet.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

small, large, step, length, radius = 0.020, 0.030, 0.120, 0.200, 0.004
shaft = Cylinder(radius=small, length=step, axis=(1.0, 0.0, 0.0)) | Cylinder(
    radius=large, length=length - step, origin=(step, 0.0, 0.0), axis=(1.0, 0.0, 0.0)
)
shaft = shaft.fillet(
    radius,
    edges=shaft.select(
        edges=lambda edge: abs(np.asarray(edge.start)[0] - step) < 1e-9
        and abs(np.linalg.norm(np.asarray(edge.start)[1:]) - small) < 1e-9
    ),
)
fillet = lambda part: part.select(
    faces=lambda face: face.kind not in ("plane", "cylinder")
)
whole = shaft.grouped(
    loaded=shaft.select(faces=lambda face: abs(face.center[0]) < 1e-9),
    clamped=shaft.select(faces=lambda face: abs(face.center[0] - length) < 1e-9),
    fillet=fillet(shaft),
)
whole.save("shaft_fillet_submodel.step")
local = shaft & Box(length=0.040, width=0.1, height=0.1, origin=(0.100, -0.05, -0.05))
local = local.grouped(
    cut=local.select(
        faces=lambda face: face.kind == "plane"
        and min(abs(face.center[0] - 0.100), abs(face.center[0] - 0.140)) < 1e-9
    ),
    fillet=fillet(local),
)
local.save("shaft_fillet_submodel_local.step")
