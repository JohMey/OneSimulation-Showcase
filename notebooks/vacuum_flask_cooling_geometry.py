"""Geometry of the OneSimulation example "Natural convection cooling of a vacuum flask".

Builds a vacuum flask holding coffee (radius 40 mm, 200 mm high) with the CAD objects of
OneSimulation: an inner glass vessel (1.5 mm), a vacuum gap (4 mm), an outer steel shell (1 mm), a
plastic shoulder ring (1.5 mm, the part neck) that joins them at the top and a plastic stopper (35 mm thick). Every
part is half a solid of revolution about the z axis, drawn as its meridian profile; the cut with
the plane y = 0 is the meridian section. Writes

- vacuum_flask_cooling.step: the flask without the coffee, with the face groups coffee (the
  surfaces wetted by the coffee), side (outer side of the shell), top (neck ring and stopper in
  the air), base (standing on the floor) and axis,
- vacuum_flask_cooling_glass.step, _vacuum.step, _shell.step, _neck.step, _stopper.step: the
  material regions; their faces on the outside of the flask carry the same names, the faces
  between two regions are called inside.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Revolution

mm = 1e-3
r1, r2, r3, r4 = 40 * mm, 41.5 * mm, 45.5 * mm, 46.5 * mm
z1, z2, z3, height, neck, plug = (
    -1.5 * mm,
    -5.5 * mm,
    -6.5 * mm,
    200 * mm,
    201.5 * mm,
    235 * mm,
)
revolved = lambda points, names=None: Revolution(
    profile=Profile.polygon(points, names), angle=np.pi
).rotated((0, 0, 1), -np.pi / 2)

regions = {
    "glass": (
        [(0, z1), (r2, z1), (r2, height), (r1, height), (r1, 0), (0, 0)],
        ["inside", "inside", "inside", "coffee", "coffee", "axis"],
    ),
    "vacuum": (
        [(0, z2), (r3, z2), (r3, height), (r2, height), (r2, z1), (0, z1)],
        ["inside", "inside", "inside", "inside", "inside", "axis"],
    ),
    "shell": (
        [(0, z3), (r4, z3), (r4, height), (r3, height), (r3, z2), (0, z2)],
        ["base", "side", "inside", "inside", "inside", "axis"],
    ),
    "neck": (
        [(r1, height), (r4, height), (r4, neck), (r1, neck)],
        ["inside", "side", "top", "inside"],
    ),
    "stopper": (
        [(0, height), (r1, height), (r1, neck), (r1, plug), (0, plug)],
        ["coffee", "inside", "top", "top", "axis"],
    ),
}
for name, (points, names) in regions.items():
    revolved(points, names).save(f"vacuum_flask_cooling_{name}.step")

revolved(
    [
        (0, z3),
        (r4, z3),
        (r4, neck),
        (r1, neck),
        (r1, plug),
        (0, plug),
        (0, height),
        (r1, height),
        (r1, 0),
        (0, 0),
    ],
    ["base", "side", "top", "top", "top", "axis", "coffee", "coffee", "coffee", "axis"],
).save("vacuum_flask_cooling.step")
