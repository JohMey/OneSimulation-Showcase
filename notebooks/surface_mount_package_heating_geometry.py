"""Geometry of the OneSimulation example "Heat transfer in a surface-mount package".

Builds a 16-pin small-outline package (SOIC-16) on a circuit board next to the footprint of a voltage
regulator with the CAD objects of OneSimulation and writes all parts as one STEP assembly,
surface_mount_package_heating.step, with the named parts board, footprint, ground_plane, package,
die, strip and lead_1 ... lead_16. The footprint keeps its top face as the group regulator.
Dimensions in mm, written in m.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Assembly, Box, Prism

mm = 1e-3
box = lambda x0, x1, y0, y1, z0, z1: Box(
    length=(x1 - x0) * mm,
    width=(y1 - y0) * mm,
    height=(z1 - z0) * mm,
    origin=(x0 * mm, y0 * mm, z0 * mm),
)
footprint = box(-10.0, -8.0, -5.0, -1.0, -1.0, -0.9)
footprint = footprint.grouped(regulator=footprint.boundaries.z_max)
ground_plane = box(-8.0, -4.0, -5.0, -1.0, -1.0, -0.9)
board = box(-10.0, 10.0, -5.0, 5.0, -1.9, -0.9) - footprint - ground_plane
die = box(-0.5, 0.5, -0.5, 0.5, -0.05, 0.05)
strip = box(-4.645, -4.245, -1.95, 0.2, -0.025, 0.025) | box(
    -4.245, -0.5, -0.2, 0.2, -0.025, 0.025
)
body = Prism(
    profile=Profile.polygon(
        [
            (y * mm, z * mm)
            for y, z in (
                (-1.95, -0.1),
                (-1.72, -0.75),
                (1.72, -0.75),
                (1.95, -0.1),
                (1.95, 0.1),
                (1.72, 0.75),
                (-1.72, 0.75),
                (-1.95, 0.1),
            )
        ]
    ),
    height=9.9 * mm,
    axis=(1.0, 0.0, 0.0),
    origin=(-4.95 * mm, 0.0, 0.0),
)
package = body - die - strip
gull_wing = (
    (-1.95, 0.1),
    (-2.55, 0.1),
    (-2.55, -0.7),
    (-2.85, -0.7),
    (-2.85, -0.9),
    (-2.35, -0.9),
    (-2.35, -0.1),
    (-1.95, -0.1),
)
leads = [
    Prism(
        profile=Profile.polygon(
            [(side * y * mm, z * mm) for y, z in gull_wing[::side]]
        ),
        height=0.4 * mm,
        axis=(1.0, 0.0, 0.0),
        origin=((-4.645 + 1.27 * k) * mm, 0.0, 0.0),
    ).named(f"lead_{k + 1 if side > 0 else 16 - k}")
    for k in range(8)
    for side in (1, -1)
]
assembly = Assembly(
    parts=(
        board.named("board"),
        footprint.named("footprint"),
        ground_plane.named("ground_plane"),
        package.named("package"),
        die.named("die"),
        strip.named("strip"),
        *leads,
    )
)
assembly.save("surface_mount_package_heating.step")
