"""Geometry of the OneSimulation example "Non-proportional fatigue of a shaft shoulder".

Builds a stepped steel shaft with the CAD objects of OneSimulation: 40 mm diameter over the first
120 mm, 60 mm diameter over the last 80 mm, joined by a shoulder fillet of 4 mm radius; the axis is
x. The faces loaded (x = 0), clamped (x = 200 mm) and fillet are named and the part is written to
shaft_fillet_nonproportional_fatigue.step.
"""

import numpy as np
from onesim.cad.d3 import Cylinder

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
shaft = shaft.grouped(
    loaded=shaft.select(faces=lambda face: abs(face.center[0]) < 1e-9),
    clamped=shaft.select(faces=lambda face: abs(face.center[0] - length) < 1e-9),
    fillet=shaft.select(faces=lambda face: face.kind not in ("plane", "cylinder")),
)
shaft.save("shaft_fillet_nonproportional_fatigue.step")
