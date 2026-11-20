"""Geometry of the OneSimulation example "Beam cross-section calculator".

Builds three short prisms with standard steel cross-sections, with the CAD objects of OneSimulation:
an IPE 200 I-beam with its 12 mm root fillets, a welded box 120 x 60 x 5 mm and a lipless channel
140 x 60 x 4 mm. Each prism is 10 mm long along z and is written to its own STEP file,
beam_section_properties_<name>.step; the example computes on their mid-plane sections.
"""

from onesim.cad.d3 import Box, UnionFromObjects

depth = 0.010
h, b, tw, tf, r = 0.200, 0.100, 0.0056, 0.0085, 0.012
ipe = UnionFromObjects(
    objects=[
        Box(length=b, width=tf, height=depth, origin=(-b / 2, h / 2 - tf, 0.0)),
        Box(length=b, width=tf, height=depth, origin=(-b / 2, -h / 2, 0.0)),
        Box(
            length=tw,
            width=h - 2 * tf,
            height=depth,
            origin=(-tw / 2, -h / 2 + tf, 0.0),
        ),
    ]
)
ipe = ipe.fillet(
    r,
    edges=ipe.select(
        edges=lambda edge: abs(edge.start[2] - edge.end[2]) > 1e-9
        and abs(abs(edge.start[0]) - tw / 2) < 1e-9
        and abs(abs(edge.start[1]) - (h / 2 - tf)) < 1e-9
    ),
)
ipe.save("beam_section_properties_ipe.step")

box = Box(length=0.120, width=0.060, height=depth, origin=(-0.060, -0.030, 0.0)) - Box(
    length=0.110, width=0.050, height=3 * depth, origin=(-0.055, -0.025, -depth)
)
box.save("beam_section_properties_box.step")

t, width, height = 0.004, 0.060, 0.140
channel = UnionFromObjects(
    objects=[
        Box(length=t, width=height, height=depth, origin=(0.0, -height / 2, 0.0)),
        Box(length=width - t, width=t, height=depth, origin=(t, height / 2 - t, 0.0)),
        Box(length=width - t, width=t, height=depth, origin=(t, -height / 2, 0.0)),
    ]
)
channel.save("beam_section_properties_channel.step")
