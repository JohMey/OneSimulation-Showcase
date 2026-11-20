"""Geometry of the OneSimulation example "Helical spring driven by a global displacement constraint".

Builds a helical compression spring (wire diameter 6 mm, mean coil diameter 40 mm, pitch 12 mm, six
turns, axis along z) as a circle swept along a helix with the CAD objects of OneSimulation, names
the two wire end faces and the wire surface, and writes loaded_spring_global_constraint.step.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Helix, Sweep

spring = Sweep(
    profile=Profile.circle(0.003), path=Helix(radius=0.020, pitch=0.012, turns=6)
)
spring = spring.grouped(
    fixed_end=spring.select(
        faces=lambda face: face.kind == "plane" and face.center[2] < 0.036
    ),
    loaded_end=spring.select(
        faces=lambda face: face.kind == "plane" and face.center[2] > 0.036
    ),
    wire=spring.select(faces=lambda face: face.kind != "plane"),
)
spring.save("loaded_spring_global_constraint.step")
