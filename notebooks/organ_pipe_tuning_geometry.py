"""Geometry of the OneSimulation example "Tuning an open organ pipe".

Builds the air in and around an open cylindrical organ pipe - inner radius 25 mm, wall 1 mm, length
625 mm, tuned to middle C - inside a sphere of air, as half a solid of revolution about the z axis with
the CAD objects of OneSimulation. The sphere has a radius of 0.57 m in organ_pipe_tuning.step and of
1.1 m in organ_pipe_tuning_large.step; both name the faces pipe_wall and outer.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

radius, wall, length = 0.025, 0.001, 0.625
half = dict(angle=np.pi)
tube = Revolution(
    profile=Profile(
        Path(
            (radius, -length / 2),
            (
                Segment((radius + wall, -length / 2), name="pipe"),
                Segment((radius + wall, length / 2), name="pipe"),
                Segment((radius, length / 2), name="pipe"),
                Segment((radius, -length / 2), name="pipe"),
            ),
        )
    ),
    **half,
)
for name, outer in (("organ_pipe_tuning", 0.57), ("organ_pipe_tuning_large", 1.1)):
    sphere = Revolution(
        profile=Profile(
            Path(
                (0.0, -outer),
                (
                    Segment((outer, 0.0), center=(0.0, 0.0), name="outer"),
                    Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
                    Segment((0.0, -outer), name="axis"),
                ),
            )
        ),
        **half,
    )
    air = sphere - tube
    air = air.grouped(
        pipe_wall=air.select(
            faces=lambda face: face.kind != "sphere" and abs(face.center[0]) > 1e-6
        )
    )
    air.save(f"{name}.step")
