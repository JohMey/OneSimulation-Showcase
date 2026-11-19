"""Geometry of the OneSimulation example "Stresses and strains in a wrench".

Builds a 13 mm combination wrench that grips the hexagon head of an M8 bolt with the CAD objects
of OneSimulation, names the faces the simulation refers to and writes wrench_stress_analysis.step.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Cylinder, Prism, UnionFromObjects

mm = 1e-3
thickness, head_radius, across_flats = 5 * mm, 16 * mm, 13 * mm
throat, throat_fillet, jaw_angle = 8.5 * mm, 2 * mm, np.radians(15)
length, handle_height, ring_radius, bore_radius, neck_fillet = (
    140 * mm,
    11 * mm,
    10.5 * mm,
    7 * mm,
    8 * mm,
)
shank_radius, shank_length = 4 * mm, 15 * mm
low = -thickness / 2


def upright(edge, x=None, y=None):
    """True for an edge along z, optionally at the given x or |y|."""
    start, end = np.asarray(edge.start), np.asarray(edge.end)
    return (
        np.allclose(start[:2], end[:2], atol=1e-9)
        and (x is None or abs(start[0] - x) < 1e-9)
        and (y is None or abs(abs(start[1]) - y) < 1e-9)
    )


head = Cylinder(radius=head_radius, length=thickness, origin=(0, 0, low))
slot = Box(
    length=3 * head_radius + throat,
    width=across_flats,
    height=2 * thickness,
    origin=(-3 * head_radius, -across_flats / 2, -thickness),
)
jaws = head - slot
jaws = jaws.fillet(
    throat_fillet, edges=jaws.select(edges=lambda edge: upright(edge, x=throat))
)

corners = [
    (
        across_flats / np.sqrt(3) * np.cos(k * np.pi / 3),
        across_flats / np.sqrt(3) * np.sin(k * np.pi / 3),
    )
    for k in range(6)
]
hexagon = Prism(profile=Profile.polygon(corners), height=thickness, origin=(0, 0, low))
shank = Cylinder(
    radius=shank_radius, length=shank_length, origin=(0, 0, low - shank_length)
)
open_end = UnionFromObjects(objects=[jaws, hexagon, shank]).rotated(
    (0, 0, 1), jaw_angle
)

handle = Box(
    length=length - 8 * mm - (head_radius - 3 * mm),
    width=handle_height,
    height=thickness,
    origin=(head_radius - 3 * mm, -handle_height / 2, low),
)
ring = Cylinder(
    radius=ring_radius, length=thickness, origin=(length, 0, low)
) - Cylinder(radius=bore_radius, length=2 * thickness, origin=(length, 0, -thickness))
wrench = UnionFromObjects(objects=[open_end, handle, ring])
wrench = wrench.fillet(
    neck_fillet,
    edges=wrench.select(edges=lambda edge: upright(edge, y=handle_height / 2)),
)

wrench = wrench.grouped(
    fixed=wrench.select(
        faces=lambda face: abs(face.center[2] - (low - shank_length)) < 1e-9
    ),
    hand=wrench.select(
        faces=lambda face: face.kind == "cylinder"
        and abs(face.radius - ring_radius) < 1e-9
        and face.center[0] > length
    ),
    throat=wrench.select(
        faces=lambda face: face.kind == "cylinder"
        and abs(face.radius - throat_fillet) < 1e-9
    ),
)
wrench.save("wrench_stress_analysis.step")
