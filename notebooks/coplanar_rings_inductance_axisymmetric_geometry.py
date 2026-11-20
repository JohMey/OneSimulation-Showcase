"""Geometry of the OneSimulation example "Self and mutual inductance of two coplanar rings".

Builds with the CAD objects of OneSimulation a cylinder of air of 1 m radius and 2 m height
around two concentric single-turn copper rings in the plane z = 0: a primary of 100 mm and a
secondary of 10 mm ring radius, both of wire with 1 mm radius. Every part is cut to the side x > 0,
so that its section with the plane y = 0 is the meridian half-plane. Files:
coplanar_rings_inductance_axisymmetric.step (air), coplanar_rings_inductance_axisymmetric_primary.step
and coplanar_rings_inductance_axisymmetric_secondary.step.
"""

from onesim.cad.d3 import Box, Cylinder, Torus

radius, height, wire = 1.0, 2.0, 0.001
side = Box(
    length=radius, width=2 * radius, height=height, origin=(0.0, -radius, -height / 2)
)
(Cylinder(radius=radius, length=height, origin=(0.0, 0.0, -height / 2)) & side).save(
    "coplanar_rings_inductance_axisymmetric.step"
)
for name, ring in (("primary", 0.1), ("secondary", 0.01)):
    (Torus(major_radius=ring, minor_radius=wire) & side).save(
        f"coplanar_rings_inductance_axisymmetric_{name}.step"
    )
