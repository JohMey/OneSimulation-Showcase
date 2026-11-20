"""Geometry of the OneSimulation example "Self and mutual inductance of two coplanar loops".

Builds with the CAD objects of OneSimulation one eighth of a ball of air of 0.6 m radius (x, y, z > 0)
around two concentric, coplanar single-turn loops in the plane z = 0, and writes it to
coplanar_loops_mutual_inductance.step with the groups planes (x = 0 and y = 0), midplane (z = 0) and
outer (the sphere). The loops have a square cross-section of 4 mm x 4 mm and mean radii of 50 mm
(primary) and 100 mm (secondary); they go whole to coplanar_loops_mutual_inductance_primary.step and
coplanar_loops_mutual_inductance_secondary.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Sphere

radius, side = 0.6, 0.004
cut = Box(length=radius, width=radius, height=radius).named("cut")
air = Sphere(radius=radius).named("ball") & cut
faces = air.boundaries
air = air.grouped(
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    midplane=faces["cut.z_min"],
    outer=air.select(faces=lambda face: face.id.startswith("ball")),
)
air.save("coplanar_loops_mutual_inductance.step")
for name, mean in (("primary", 0.05), ("secondary", 0.1)):
    AnnularCylinder(
        inner_radius=mean - side / 2,
        outer_radius=mean + side / 2,
        length=side,
        origin=(0.0, 0.0, -side / 2),
    ).save(f"coplanar_loops_mutual_inductance_{name}.step")
