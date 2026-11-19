"""Geometry of the OneSimulation example "Star-shaped microchannel driven by five pulsating inlets".

Builds the microchannel with the CAD objects of OneSimulation: a flat round chamber of radius
0.6 mm and depth 0.2 mm, five straight inlet channels of 0.2 mm x 0.2 mm cross-section reaching
out to a radius of 1.6 mm at 72 degree spacing, and an outlet tube of radius 0.2 mm that rises
1 mm from the centre of the chamber. The part is written to star_microchannel_superposition.step
with the face groups inlet_1 to inlet_5 (the channel ends, counter-clockwise from the x axis) and
outlet (the top of the tube).
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder, UnionFromObjects

mm = 1e-3
depth, reach = 0.2 * mm, 1.6 * mm
parts = [
    Cylinder(radius=0.6 * mm, length=depth),
    Cylinder(radius=0.2 * mm, length=1.0 * mm, origin=(0.0, 0.0, depth)),
]
parts += [
    Box(
        length=reach - 0.4 * mm,
        width=0.2 * mm,
        height=depth,
        origin=(0.4 * mm, -0.1 * mm, 0.0),
    ).rotated((0, 0, 1), np.radians(72 * k))
    for k in range(5)
]
star = UnionFromObjects(objects=parts)
end_of = lambda k: (
    lambda face: abs(face.center[0] - reach * np.cos(np.radians(72 * k))) < 1e-9
    and abs(face.center[1] - reach * np.sin(np.radians(72 * k))) < 1e-9
)
star = star.grouped(
    outlet=star.select(
        faces=lambda face: abs(face.center[2] - depth - 1.0 * mm) < 1e-9
    ),
    **{f"inlet_{k + 1}": star.select(faces=end_of(k)) for k in range(5)},
)
star.save("star_microchannel_superposition.step")
