"""Geometry of the OneSimulation example "Stresses around a multilateral well junction".

Builds half of a rock block (2 m x 1 m x 2 m, cut by the symmetry plane y = 0) around a vertical
main bore of 100 mm diameter with an inclined, 0.8 m long lateral leg of 80 mm diameter that branches off
at mid height, with the CAD objects of OneSimulation. The bore walls and the outer faces are named and the
part is written to multilateral_well_stability.step.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder, UnionFromObjects

half = Box(length=2.0, width=1.0, height=2.0, origin=(-1.0, 0.0, -1.0))
main = Cylinder(radius=0.05, length=2.4, origin=(0.0, 0.0, -1.2))
angle = np.radians(45.0)
lateral = Cylinder(
    radius=0.04,
    length=0.8,
    origin=(0.0, 0.0, 0.0),
    axis=(np.sin(angle), 0.0, -np.cos(angle)),
)
rock = half - UnionFromObjects(objects=[main, lateral])
rock = rock.grouped(
    main_bore=rock.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - 0.05) < 1e-6
    ),
    lateral_bore=rock.select(
        faces=lambda face: (face.kind == "cylinder" and abs(face.radius - 0.04) < 1e-6)
        or (
            face.kind == "plane"
            and 0.1 < face.center[0] < 0.9
            and -0.9 < face.center[2] < -0.1
        )
    ),
    symmetry=rock.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-9
    ),
    x_low=rock.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0] + 1.0) < 1e-9
    ),
    x_high=rock.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0] - 1.0) < 1e-9
    ),
    y_high=rock.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1] - 1.0) < 1e-9
    ),
    z_low=rock.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[2] + 1.0) < 1e-9
    ),
    z_high=rock.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[2] - 1.0) < 1e-9
    ),
)
rock.save("multilateral_well_stability.step")
