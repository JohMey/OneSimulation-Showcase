"""Geometry of the OneSimulation example "Acoustic cloak around a rigid cylinder".

Builds, as thin plates with the CAD objects of OneSimulation, the air around a rigid cylinder of
radius 0.1 m out to a circle of radius 0.6 m (acoustic_cloak.step, with the outer face named outer
and the cylinder face named obstacle) and the cloak, an annulus between 0.1 m and 0.2 m
(acoustic_cloak_shell.step).
"""

from onesim.cad.d3 import Cylinder

obstacle, cloak, outer, thickness = 0.1, 0.2, 0.6, 0.01
air = Cylinder(radius=outer, length=thickness) - Cylinder(
    radius=obstacle, length=thickness
)
round_face = lambda radius: (
    lambda face: face.kind == "cylinder" and abs(face.radius - radius) < 1e-9
)
air = air.grouped(
    outer=air.select(faces=round_face(outer)),
    obstacle=air.select(faces=round_face(obstacle)),
)
air.save("acoustic_cloak.step")
(
    Cylinder(radius=cloak, length=thickness)
    - Cylinder(radius=obstacle, length=thickness)
).save("acoustic_cloak_shell.step")
