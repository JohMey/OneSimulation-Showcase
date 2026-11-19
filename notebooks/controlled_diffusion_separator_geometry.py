"""Geometry of the OneSimulation example "Controlled diffusion separator".

Builds the lower half of the H-shaped microchannel with the CAD objects of OneSimulation: two inlet
and two outlet arms, 10 µm wide, each joined by a quarter bend to the contact zone, 20 µm wide and
100 µm long; the channel is 20 µm deep, the file holds the 10 µm below the mid-depth plane. Writes
controlled_diffusion_separator.step with the face groups inlet_a, inlet_b, outlet_a, outlet_b,
mid_plane and bottom.
"""

from onesim.cad.d3 import AnnularCylinder, Box, UnionFromObjects

um = 1e-6
width, half_depth, arm, inner, outer, start, end = (
    10 * um,
    10 * um,
    40 * um,
    10 * um,
    20 * um,
    20 * um,
    120 * um,
)

parts = [
    Box(
        length=end - start,
        width=2 * width,
        height=half_depth,
        origin=(start, -width, 0.0),
    )
]
for x_bend, x_arm in ((start, start - outer), (end, end + inner)):
    for side in (1, -1):
        ring = AnnularCylinder(
            inner_radius=inner,
            outer_radius=outer,
            length=half_depth,
            origin=(x_bend, side * outer, 0.0),
        )
        quadrant = Box(
            length=outer,
            width=outer,
            height=half_depth,
            origin=(
                x_bend - outer if x_bend == start else x_bend,
                0.0 if side > 0 else -outer,
                0.0,
            ),
        )
        parts += [
            ring & quadrant,
            Box(
                length=width,
                width=arm,
                height=half_depth,
                origin=(x_arm, outer if side > 0 else -outer - arm, 0.0),
            ),
        ]
channel = UnionFromObjects(objects=parts)

at_end = (
    lambda face, x, side: abs(face.center[1] - side * (outer + arm)) < 1e-9
    and abs(face.center[0] - x) < 1e-9
)
channel = channel.grouped(
    inlet_a=channel.select(
        faces=lambda face: at_end(face, start - outer + width / 2, -1)
    ),
    inlet_b=channel.select(
        faces=lambda face: at_end(face, start - outer + width / 2, 1)
    ),
    outlet_a=channel.select(
        faces=lambda face: at_end(face, end + inner + width / 2, -1)
    ),
    outlet_b=channel.select(
        faces=lambda face: at_end(face, end + inner + width / 2, 1)
    ),
    mid_plane=channel.select(
        faces=lambda face: face.kind == "plane"
        and abs(face.center[2] - half_depth) < 1e-12
    ),
    bottom=channel.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[2]) < 1e-12
    ),
)
channel.save("controlled_diffusion_separator.step")
