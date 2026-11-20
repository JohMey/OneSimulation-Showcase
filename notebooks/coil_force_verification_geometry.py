"""Geometry of the OneSimulation example "Force between two coaxial coils: a verification series".

Builds one quarter of the air around two coaxial coils with the CAD objects of OneSimulation: a
quarter of a ball of 0.4 m radius, cut by the planes x = 0 and y = 0 (face group planes; the ball
surface is outer), and writes coil_force_verification.step. The quarters of the windings (mean
radius 50 mm, 10 x 10 mm cross-section) are written to coil_force_verification_lower.step (centred
at z = 0) and coil_force_verification_upper_25mm.step, ..._50mm.step, ..._100mm.step for the three
coil distances.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Sphere

air = 0.4
quarter = Box(length=air, width=air, height=2 * air, origin=(0.0, 0.0, -air)).named(
    "cut"
)
region = Sphere(radius=air).named("ball") & quarter
faces = region.boundaries
region.grouped(
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    outer=region.select(faces=lambda face: face.id.startswith("ball")),
).save("coil_force_verification.step")
winding = (
    lambda z: AnnularCylinder(
        inner_radius=0.045,
        outer_radius=0.055,
        length=0.01,
        origin=(0.0, 0.0, z - 0.005),
    )
    & quarter
)
winding(0.0).save("coil_force_verification_lower.step")
for distance in (25, 50, 100):
    winding(distance * 1e-3).save(f"coil_force_verification_upper_{distance}mm.step")
