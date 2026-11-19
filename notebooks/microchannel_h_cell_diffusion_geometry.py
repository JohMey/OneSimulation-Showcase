"""Geometry of the OneSimulation example "Microchannel H-cell".

Builds the lower half of an H-shaped microchannel with the CAD objects of OneSimulation. Two inlet
arms (left) and two outlet arms (right), each 10 µm wide and 40 µm long, turn through quarter bends
(inner radius 10 µm, outer radius 20 µm) into the contact zone, 20 µm wide and 100 µm long. The
channel is 20 µm deep; the file holds the 10 µm below the mid-depth plane. Writes
microchannel_h_cell_diffusion.step with the face groups inlet_a, inlet_b, outlet_a, outlet_b,
mid_plane and walls (side walls and bottom).
"""

from onesim.cad.d3 import AnnularCylinder, Box, UnionFromObjects

um = 1e-6
width, half_depth, arm, inner, outer = 10 * um, 10 * um, 40 * um, 10 * um, 20 * um
start, end = 20 * um, 120 * um

parts = [
    Box(
        length=end - start,
        width=2 * width,
        height=half_depth,
        origin=(start, -width, 0.0),
    )
]
for x_bend, x_quadrant, x_arm in (
    (start, start - outer, start - outer),
    (end, end, end + inner),
):
    for side in (1, -1):
        y_quadrant = 0.0 if side > 0 else -outer
        bend = AnnularCylinder(
            inner_radius=inner,
            outer_radius=outer,
            length=half_depth,
            origin=(x_bend, side * outer, 0.0),
        )
        parts += [
            bend
            & Box(
                length=outer,
                width=outer,
                height=half_depth,
                origin=(x_quadrant, y_quadrant, 0.0),
            ),
            Box(
                length=width,
                width=arm,
                height=half_depth,
                origin=(x_arm, outer if side > 0 else -outer - arm, 0.0),
            ),
        ]
channel = UnionFromObjects(objects=parts)

port = lambda x, side: channel.select(
    faces=lambda face: abs(face.center[1] - side * (outer + arm)) < 1e-12
    and abs(face.center[0] - x - width / 2) < 1e-12
)
groups = dict(
    inlet_a=port(start - outer, -1),
    inlet_b=port(start - outer, 1),
    outlet_a=port(end + inner, -1),
    outlet_b=port(end + inner, 1),
    mid_plane=channel.select(
        faces=lambda face: face.kind == "plane"
        and abs(face.center[2] - half_depth) < 1e-12
    ),
)
ends = (
    groups["inlet_a"]
    | groups["inlet_b"]
    | groups["outlet_a"]
    | groups["outlet_b"]
    | groups["mid_plane"]
)
channel = channel.grouped(
    **groups, walls=channel.select(faces=lambda face: True) - ends
)
channel.save("microchannel_h_cell_diffusion.step")
