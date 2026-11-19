"""Geometry of the OneSimulation example "Magnetic force by virtual work and by the Maxwell stress tensor".

Builds one quarter of the air around a pair of ring magnets with the CAD objects of OneSimulation:
a quarter of a ball of 100 mm radius (face groups planes and outer), written to
magnet_force_virtual_work_maxwell.step. The lower, fixed ring magnet (inner radius 10 mm, outer
radius 20 mm, 10 mm high, top face at z = 0) goes to magnet_force_virtual_work_maxwell_lower.step.
The upper ring is shifted by -0.5 mm and +0.5 mm about a 4 mm gap; to mesh both positions alike its
space is cut into three slices written to ..._slice_bottom.step (z = 3.5 to 4.5 mm),
..._slice_middle.step (4.5 to 13.5 mm) and ..._slice_top.step (13.5 to 14.5 mm).
"""

from onesim.cad.d3 import AnnularCylinder, Box, Sphere

mm = 1e-3
air, inner, outer, height, gap, shift = (
    100 * mm,
    10 * mm,
    20 * mm,
    10 * mm,
    4 * mm,
    0.5 * mm,
)
quarter = Box(length=air, width=air, height=2 * air, origin=(0.0, 0.0, -air)).named(
    "cut"
)
region = Sphere(radius=air).named("ball") & quarter
faces = region.boundaries
region.grouped(
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    outer=region.select(faces=lambda face: face.id.startswith("ball")),
).save("magnet_force_virtual_work_maxwell.step")
ring = (
    lambda bottom, top: AnnularCylinder(
        inner_radius=inner,
        outer_radius=outer,
        length=top - bottom,
        origin=(0.0, 0.0, bottom),
    )
    & quarter
)
ring(-height, 0.0).save("magnet_force_virtual_work_maxwell_lower.step")
ring(gap - shift, gap + shift).save(
    "magnet_force_virtual_work_maxwell_slice_bottom.step"
)
ring(gap + shift, gap + height - shift).save(
    "magnet_force_virtual_work_maxwell_slice_middle.step"
)
ring(gap + height - shift, gap + height + shift).save(
    "magnet_force_virtual_work_maxwell_slice_top.step"
)
