"""Geometry of the OneSimulation example "Inductance of a gapped power inductor".

Builds one quarter of the air around a ferrite pot-core inductor with the CAD objects of
OneSimulation: a quarter of a ball of 50 mm radius, cut by the planes x = 0 and y = 0 (face group
planes; the ball surface is outer), written to power_inductor_inductance.step. The quarter of the
ferrite core (outer radius 14 mm, height 14 mm, winding window from 5 to 12 mm radius and 10 mm
high, 0.5 mm air gap in the centre post) goes to power_inductor_inductance_core.step, the quarter of
the winding (6 to 11 mm radius, 9 mm high) to power_inductor_inductance_winding.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder, Sphere

mm = 1e-3
air = 50 * mm
quarter = Box(length=air, width=air, height=2 * air, origin=(0.0, 0.0, -air)).named(
    "cut"
)
region = Sphere(radius=air).named("ball") & quarter
faces = region.boundaries
region.grouped(
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    outer=region.select(faces=lambda face: face.id.startswith("ball")),
).save("power_inductor_inductance.step")
core = (
    Cylinder(radius=14 * mm, length=14 * mm, origin=(0.0, 0.0, -7 * mm))
    - AnnularCylinder(
        inner_radius=5 * mm,
        outer_radius=12 * mm,
        length=10 * mm,
        origin=(0.0, 0.0, -5 * mm),
    )
    - Cylinder(radius=5 * mm, length=0.5 * mm, origin=(0.0, 0.0, -0.25 * mm))
)
(core & quarter).save("power_inductor_inductance_core.step")
(
    AnnularCylinder(
        inner_radius=6 * mm,
        outer_radius=11 * mm,
        length=9 * mm,
        origin=(0.0, 0.0, -4.5 * mm),
    )
    & quarter
).save("power_inductor_inductance_winding.step")
