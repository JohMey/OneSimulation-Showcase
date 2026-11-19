"""Geometry of the OneSimulation example "Induction currents from a circular coil in a copper ring".

Builds, with the CAD objects of OneSimulation, a circular coil (100 turns, winding section 10 mm x
10 mm, inner radius 40 mm, centred 20 mm above the midplane) and a coaxial copper ring (square
section 6 mm x 6 mm, inner radius 40 mm, centred on the midplane) inside a cylinder of air of
0.3 m radius and 0.6 m height. Each part is cut to the side x > 0, so that its section with the
plane y = 0 is the meridian half-plane. Files: induction_currents_coil_and_ring.step (the air, the
domain), induction_currents_coil_and_ring_coil.step and induction_currents_coil_and_ring_ring.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

extent = 0.3
side = Box(
    length=extent, width=2 * extent, height=2 * extent, origin=(0.0, -extent, -extent)
)
(Cylinder(radius=extent, length=2 * extent, origin=(0.0, 0.0, -extent)) & side).save(
    "induction_currents_coil_and_ring.step"
)
(
    AnnularCylinder(
        inner_radius=0.040, outer_radius=0.050, length=0.010, origin=(0.0, 0.0, 0.015)
    )
    & side
).save("induction_currents_coil_and_ring_coil.step")
(
    AnnularCylinder(
        inner_radius=0.040, outer_radius=0.046, length=0.006, origin=(0.0, 0.0, -0.003)
    )
    & side
).save("induction_currents_coil_and_ring_ring.step")
