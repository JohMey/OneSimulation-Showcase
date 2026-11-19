"""Geometry of the OneSimulation example "Submodel of a vent hole in a steel wheel".

Builds a pressed steel wheel with the CAD objects of OneSimulation: a disc 6 mm thick and 336 mm in
diameter with a 60 mm bore, five bolt holes of 14 mm on a 100 mm circle and five vent holes of
50 mm on a 220 mm circle (one of them at the bottom), joined to a rim 340 mm in diameter, 5 mm
thick and 140 mm wide. The wheel axis is z. The whole wheel is written to wheel_rim_submodel.step
with the face groups bolts, rim (outer rim surface) and vent (the bottom vent hole). The submodel,
the part of the disc around the bottom vent hole (|x| <= 45 mm, -155 mm <= y <= -65 mm), is written
to wheel_rim_submodel_local.step with the face groups cut (the four cut planes) and vent.
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

disc = Cylinder(radius=0.168, length=0.006)
disc = disc.hole(0.060, at=(0.0, 0.0, 0.006))
for k in range(5):
    bolt, vent = np.radians(-90 + 36 + 72 * k), np.radians(-90 + 72 * k)
    disc = disc.hole(0.014, at=(0.050 * np.cos(bolt), 0.050 * np.sin(bolt), 0.006))
    disc = disc.hole(0.050, at=(0.110 * np.cos(vent), 0.110 * np.sin(vent), 0.006))
wheel = disc | AnnularCylinder(
    inner_radius=0.165, outer_radius=0.170, length=0.140, origin=(0.0, 0.0, -0.100)
)
bottom_vent = lambda part: part.select(
    faces=lambda face: face.kind == "cylinder"
    and abs(face.radius - 0.025) < 1e-9
    and face.center[1] < -0.09
)
whole = wheel.grouped(
    bolts=wheel.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - 0.007) < 1e-9
    ),
    rim=wheel.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - 0.170) < 1e-9
    ),
    vent=bottom_vent(wheel),
)
whole.save("wheel_rim_submodel.step")
local = wheel & Box(
    length=0.090, width=0.090, height=0.030, origin=(-0.045, -0.155, -0.012)
)
local = local.grouped(
    cut=local.select(
        faces=lambda face: face.kind == "plane" and abs(face.normal[2]) < 0.5
    ),
    vent=bottom_vent(local),
)
local.save("wheel_rim_submodel_local.step")
