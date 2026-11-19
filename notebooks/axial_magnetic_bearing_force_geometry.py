"""Geometry of the OneSimulation example "Axial magnetic bearing with permanent magnets".

Builds one quarter of the air around a pair of ring magnets with the CAD objects of OneSimulation:
a quarter of a ball of 100 mm radius, cut by the planes x = 0 and y = 0 (face group planes; the
ball surface is outer), written to axial_magnetic_bearing_force.step. The quarter of the lower,
fixed ring magnet (inner radius 10 mm, outer radius 20 mm, 10 mm high, top face at z = 0) goes to
axial_magnetic_bearing_force_lower.step, the upper ring for the air gaps 2, 4 and 8 mm to
axial_magnetic_bearing_force_upper_2mm.step, ..._4mm.step and ..._8mm.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Sphere

mm = 1e-3
air, inner, outer, height = 100 * mm, 10 * mm, 20 * mm, 10 * mm
quarter = Box(length=air, width=air, height=2 * air, origin=(0.0, 0.0, -air)).named(
    "cut"
)
region = Sphere(radius=air).named("ball") & quarter
faces = region.boundaries
region.grouped(
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    outer=region.select(faces=lambda face: face.id.startswith("ball")),
).save("axial_magnetic_bearing_force.step")
(
    AnnularCylinder(
        inner_radius=inner,
        outer_radius=outer,
        length=height,
        origin=(0.0, 0.0, -height),
    )
    & quarter
).save("axial_magnetic_bearing_force_lower.step")
for gap in (2, 4, 8):
    (
        AnnularCylinder(
            inner_radius=inner,
            outer_radius=outer,
            length=height,
            origin=(0.0, 0.0, gap * mm),
        )
        & quarter
    ).save(f"axial_magnetic_bearing_force_upper_{gap}mm.step")
