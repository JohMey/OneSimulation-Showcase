"""Geometry of the OneSimulation example "Helical static mixer: mixing element by element".

Builds a pipe of diameter 4 mm with six helical mixing elements with the CAD objects of
OneSimulation: blades 0.4 mm thick and 6 mm long, each twisted by 180 degrees, alternately to the
right and to the left, every blade turned by 90 degrees against the previous one, with 4 mm of empty
pipe before and after them (44 mm in all). The fluid region is written to
helical_static_mixer_elements.step with the face groups inlet (z = 0) and outlet (z = 44 mm).
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Cylinder

mm = 1e-3
radius, thickness, element, count = 2 * mm, 0.4 * mm, 6 * mm, 6
length = 8 * mm + count * element
blade = Profile.rectangle(
    2.4 * radius, thickness, origin=(-1.2 * radius, -thickness / 2)
)
pipe = Cylinder(radius=radius, length=length)
for k in range(count):
    pipe = pipe - blade.twisted(element, np.pi * (-1) ** k).rotated(
        (0, 0, 1), k * np.pi / 2
    ).moved((0.0, 0.0, 4 * mm + k * element))
pipe = pipe.grouped(
    inlet=pipe.select(faces=lambda face: abs(face.center[2]) < 1e-9),
    outlet=pipe.select(faces=lambda face: abs(face.center[2] - length) < 1e-9),
)
pipe.save("helical_static_mixer_elements.step")
