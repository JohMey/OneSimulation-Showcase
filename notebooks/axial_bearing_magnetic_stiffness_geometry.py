"""Geometry of the OneSimulation example "Magnetic stiffness of an axial permanent-magnet bearing".

Builds one half of the air around a pair of ring magnets with the CAD objects of OneSimulation: a
half ball of 100 mm radius on the side y > 0, cut by the plane y = 0 (face group plane; the ball
surface is outer), written to axial_bearing_magnetic_stiffness.step. The half of the lower, fixed
ring magnet (inner radius 10 mm, outer radius 20 mm, 10 mm high, top face at z = 0) goes to
axial_bearing_magnetic_stiffness_lower.step. The upper ring is written for three positions about
the 4 mm gap: 0.5 mm lower (..._upper_down.step), 0.5 mm higher (..._upper_up.step) and 0.5 mm off
the axis in x (..._upper_side.step).
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
half = Box(length=2 * air, width=air, height=2 * air, origin=(-air, 0.0, -air)).named(
    "cut"
)
region = Sphere(radius=air).named("ball") & half
faces = region.boundaries
region.grouped(
    plane=faces["cut.y_min"],
    outer=region.select(faces=lambda face: face.id.startswith("ball")),
).save("axial_bearing_magnetic_stiffness.step")
ring = (
    lambda x, z: AnnularCylinder(
        inner_radius=inner, outer_radius=outer, length=height, origin=(x, 0.0, z)
    )
    & half
)
ring(0.0, -height).save("axial_bearing_magnetic_stiffness_lower.step")
for name, (x, z) in {
    "down": (0.0, gap - shift),
    "up": (0.0, gap + shift),
    "side": (shift, gap),
}.items():
    ring(x, z).save(f"axial_bearing_magnetic_stiffness_upper_{name}.step")
