"""Geometry of the OneSimulation example "Singular stress at a re-entrant corner".

Builds an L-shaped plate, 100 mm x 100 mm with legs 50 mm wide and 5 mm thick, with the CAD
objects of OneSimulation. reentrant_corner_singular_stress.step has the sharp re-entrant corner at
(50, 50) mm; reentrant_corner_singular_stress_fillet.step is the same plate with the corner rounded
to a radius of 2 mm. The faces are named fixed (x = 100 mm), loaded (y = 100 mm) and corner (the two
faces that meet at the re-entrant corner, with the fillet between them).
"""

import numpy as np
from onesim.cad.d3 import Box

leg, length, thickness, radius = 0.050, 0.100, 0.005, 0.002
for suffix, fillet in (("", None), ("_fillet", radius)):
    plate = Box(length=length, width=length, height=thickness) - Box(
        length=leg, width=leg, height=thickness, origin=(leg, leg, 0.0)
    )
    if fillet:
        plate = plate.fillet(
            fillet,
            edges=plate.select(
                edges=lambda edge: np.allclose(np.asarray(edge.start)[:2], (leg, leg))
                and np.allclose(np.asarray(edge.end)[:2], (leg, leg))
            ),
        )
    corner = plate.select(
        faces=lambda face: (face.kind == "cylinder")
        or (
            face.kind == "plane"
            and abs(face.normal[2]) < 0.5
            and (abs(face.center[0] - leg) < 1e-9 or abs(face.center[1] - leg) < 1e-9)
            and face.center[0] > leg / 2
            and face.center[1] > leg / 2
        )
    )
    plate = plate.grouped(
        fixed=plate.boundaries.x_max, loaded=plate.boundaries.y_max, corner=corner
    )
    plate.save(f"reentrant_corner_singular_stress{suffix}.step")
