"""Geometry of the OneSimulation example "Isospectral drums".

Builds the two drumheads of Gordon, Webb and Wolpert as thin plates with the CAD objects of
OneSimulation: polygons with eight sides, each made of seven right isosceles triangles with legs of
0.2 m. The first drum is written to isospectral_drums.step, the second to
isospectral_drums_twin.step; in both files the side faces are named rim.
"""

from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Prism

unit, thickness = 0.1, 0.01
outlines = {
    "isospectral_drums": [
        (-3, -3),
        (-3, -1),
        (1, 3),
        (1, 1),
        (3, 1),
        (1, -1),
        (-1, -1),
        (-1, -3),
    ],
    "isospectral_drums_twin": [
        (-3, 1),
        (1, 1),
        (1, 3),
        (3, 1),
        (1, -1),
        (-1, -1),
        (-1, -3),
        (-3, -1),
    ],
}
for name, corners in outlines.items():
    points = [(unit * x, unit * y) for x, y in corners]
    drum = Prism(
        profile=Profile(
            Path(points[0], tuple(Segment(p) for p in points[1:] + points[:1]))
        ),
        height=thickness,
    )
    drum = drum.grouped(
        rim=drum.select(
            faces=lambda face: face.kind == "plane" and abs(face.normal[2]) < 1e-9
        )
    )
    drum.save(f"{name}.step")
