"""Geometry of the OneSimulation example "Mutual inductance of a ring and a coil of 20 resolved turns".

Builds with the CAD objects of OneSimulation a cylinder of air of 1 m radius and 2 m height
around a single-turn copper ring of 100 mm radius (wire radius 1 mm) in the plane z = 0 and, at its
centre, a coil of 20 turns of the same wire: two layers of ten turns, turn centres at radii 8.5 mm
and 11.5 mm and every 3 mm from z = -13.5 mm to 13.5 mm, each turn a torus. Every part is cut to the
side x > 0, so that its section with the plane y = 0 is the meridian half-plane. Files:
explicit_turns_coil_mutual_inductance.step (air), explicit_turns_coil_mutual_inductance_ring.step
and explicit_turns_coil_mutual_inductance_turns.step (the 20 turns as one body of separate pieces).
"""

from onesim.cad.d3 import Box, Cylinder, Torus

radius, height, wire = 1.0, 2.0, 0.001
side = Box(
    length=radius, width=2 * radius, height=height, origin=(0.0, -radius, -height / 2)
)
(Cylinder(radius=radius, length=height, origin=(0.0, 0.0, -height / 2)) & side).save(
    "explicit_turns_coil_mutual_inductance.step"
)
(Torus(major_radius=0.1, minor_radius=wire) & side).save(
    "explicit_turns_coil_mutual_inductance_ring.step"
)
turns = None
for layer in (0.0085, 0.0115):
    for index in range(10):
        turn = (
            Torus(
                major_radius=layer,
                minor_radius=wire,
                origin=(0.0, 0.0, -0.0135 + 0.003 * index),
            )
            & side
        )
        turns = turn if turns is None else turns | turn
turns.save("explicit_turns_coil_mutual_inductance_turns.step")
