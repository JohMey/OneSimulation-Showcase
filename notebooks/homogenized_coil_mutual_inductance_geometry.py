"""Geometry of the OneSimulation example "Mutual inductance of a ring and a homogenized 20-turn coil".

Builds with the CAD objects of OneSimulation a cylinder of air of 1 m radius and 2 m height
around a single-turn copper ring of 100 mm radius (wire radius 1 mm) in the plane z = 0 and, at its
centre, the winding space of a 20-turn coil: two layers of ten turns of 1 mm wire radius, turn
centres at radii 8.5 mm and 11.5 mm and every 3 mm from z = -13.5 mm to 13.5 mm, held as one block
from r = 7.5 mm to 12.5 mm and z = -14.5 mm to 14.5 mm. Every part is cut to the side x > 0, so that
its section with the plane y = 0 is the meridian half-plane. Files:
homogenized_coil_mutual_inductance.step (air), homogenized_coil_mutual_inductance_ring.step and
homogenized_coil_mutual_inductance_coil.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder, Torus

radius, height = 1.0, 2.0
side = Box(
    length=radius, width=2 * radius, height=height, origin=(0.0, -radius, -height / 2)
)
(Cylinder(radius=radius, length=height, origin=(0.0, 0.0, -height / 2)) & side).save(
    "homogenized_coil_mutual_inductance.step"
)
(Torus(major_radius=0.1, minor_radius=0.001) & side).save(
    "homogenized_coil_mutual_inductance_ring.step"
)
(
    AnnularCylinder(
        inner_radius=0.0075,
        outer_radius=0.0125,
        length=0.029,
        origin=(0.0, 0.0, -0.0145),
    )
    & side
).save("homogenized_coil_mutual_inductance_coil.step")
