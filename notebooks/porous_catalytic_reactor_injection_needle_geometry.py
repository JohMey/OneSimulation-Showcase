"""Geometry of the OneSimulation example "Porous catalytic reactor with an injection needle".

Builds a tubular reactor, 20 mm in diameter and 60 mm long, filled with a porous catalyst bed, and
the injection needle (1.2 mm in diameter) that enters through the side wall 20 mm behind the inlet
and ends on the reactor axis, with the CAD objects of OneSimulation. The needle is cut out of the
bed; the faces inlet, outlet, wall, needle_tip and needle_wall are named and the part is written to
porous_catalytic_reactor_injection_needle.step.
"""

import numpy as np
from onesim.cad.d3 import Cylinder

mm = 1e-3
radius, length, needle_radius, needle_position = 10 * mm, 60 * mm, 0.6 * mm, 20 * mm
tube = Cylinder(radius=radius, length=length)
needle = Cylinder(radius=needle_radius, length=radius + 2 * mm).rotated((0, 1, 0), np.pi / 2).moved((0, 0, needle_position))
bed = tube - needle
bed = bed.grouped(
    inlet=bed.select(faces=lambda face: face.kind == "plane" and abs(face.center[2]) < 1e-9),
    outlet=bed.select(faces=lambda face: face.kind == "plane" and abs(face.center[2] - length) < 1e-9),
    wall=bed.select(faces=lambda face: face.kind == "cylinder" and abs(face.radius - radius) < 1e-9),
    needle_tip=bed.select(faces=lambda face: face.kind == "plane" and abs(face.center[0]) < 1e-9 and abs(face.center[2] - needle_position) < 1e-6),
    needle_wall=bed.select(faces=lambda face: face.kind == "cylinder" and abs(face.radius - needle_radius) < 1e-9),
)
bed.save("porous_catalytic_reactor_injection_needle.step")
