"""Geometry of the OneSimulation example "Laminar static mixer with twisted blades".

Builds a pipe of diameter 4 mm and length 20 mm with two mixing elements of the Kenics type with
the CAD objects of OneSimulation: blades 0.4 mm thick and 6 mm long, each twisted by 180 degrees,
the first to the right and the second to the left and turned by 90 degrees against the first. An
empty pipe of 4 mm stays before and after the elements. The fluid region is written to
kenics_static_mixer.step with the face groups inlet (z = 0) and outlet (z = 20 mm); all other faces
are walls (pipe wall and blades). The same pipe without blades is written to
kenics_static_mixer_empty.step for comparison.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Cylinder

mm = 1e-3
radius, thickness, element, length = 2 * mm, 0.4 * mm, 6 * mm, 20 * mm
blade = Profile.rectangle(
    2.4 * radius, thickness, origin=(-1.2 * radius, -thickness / 2)
)
first = blade.twisted(element, np.pi).moved((0.0, 0.0, 4 * mm))
second = (
    blade.twisted(element, -np.pi)
    .rotated((0, 0, 1), np.pi / 2)
    .moved((0.0, 0.0, 4 * mm + element))
)
pipe = Cylinder(radius=radius, length=length) - first - second
pipe = pipe.grouped(
    inlet=pipe.select(faces=lambda face: abs(face.center[2]) < 1e-9),
    outlet=pipe.select(faces=lambda face: abs(face.center[2] - length) < 1e-9),
)
pipe.save("kenics_static_mixer.step")
empty = Cylinder(radius=radius, length=length)
empty = empty.grouped(
    inlet=empty.select(faces=lambda face: abs(face.center[2]) < 1e-9),
    outlet=empty.select(faces=lambda face: abs(face.center[2] - length) < 1e-9),
)
empty.save("kenics_static_mixer_empty.step")
