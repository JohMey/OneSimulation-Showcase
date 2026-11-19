"""Geometry of the OneSimulation example "Heat transfer through a finned pipe".

Builds one periodic element of the finned copper pipe with the CAD objects of OneSimulation and
writes finned_pipe_heat_transfer.step: one fin pitch (5 mm) along the pipe axis x and a 15 degree
wedge from a groove ridge to a groove valley. Pipe: 20 mm bore, 2 mm wall, 12 grooves 0.8 mm deep
(a cubic Bezier curve between ridge and valley); fin: 0.6 mm thick, 35 mm outer radius. The file
names the face groups bore, pipe_surface, fin_surface, end_planes and wedge_planes.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Prism, Revolution, UnionFromObjects

mm = 1e-3
bore, wall, fin, fin_thickness, pitch, grooves, depth = (
    10 * mm,
    12 * mm,
    35 * mm,
    0.6 * mm,
    5 * mm,
    12,
    0.8 * mm,
)
wedge = np.pi / grooves
at = lambda radius, angle: (radius * np.cos(angle), radius * np.sin(angle))
ridge, valley = at(bore + depth, 0.0), at(bore, wedge)
chord = np.hypot(valley[0] - ridge[0], valley[1] - ridge[1])
groove = Segment(
    valley,
    control=(
        (ridge[0], 0.333 * chord),
        (
            valley[0] + 0.807 * chord * np.sin(wedge),
            valley[1] - 0.807 * chord * np.cos(wedge),
        ),
    ),
)
wall_section = Profile(
    Path(
        ridge,
        (
            groove,
            Segment(at(wall, wedge)),
            Segment((wall, 0.0), center=(0.0, 0.0), counterclockwise=False),
            Segment(ridge),
        ),
    )
)
fin_section = Profile.polygon(
    [
        (wall, -fin_thickness / 2),
        (fin, -fin_thickness / 2),
        (fin, fin_thickness / 2),
        (wall, fin_thickness / 2),
    ]
)
pipe = UnionFromObjects(
    objects=[
        Prism(
            profile=wall_section,
            height=pitch,
            axis=(1.0, 0.0, 0.0),
            origin=(-pitch / 2, 0.0, 0.0),
        ),
        Revolution(profile=fin_section, angle=wedge, axis=(1.0, 0.0, 0.0)),
    ]
)

radius = lambda face: np.hypot(face.center[1], face.center[2])
across = (
    lambda face, x: face.kind == "plane"
    and abs(abs(face.normal[0]) - 1) < 1e-9
    and abs(abs(face.center[0]) - x) < 1e-9
)
pipe = pipe.grouped(
    bore=pipe.select(
        faces=lambda face: face.kind != "plane" and radius(face) < wall - 1 * mm
    ),
    pipe_surface=pipe.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - wall) < 1e-9
    ),
    fin_surface=pipe.select(
        faces=lambda face: across(face, fin_thickness / 2)
        or (face.kind == "cylinder" and abs(face.radius - fin) < 1e-9)
    ),
    end_planes=pipe.select(faces=lambda face: across(face, pitch / 2)),
    wedge_planes=pipe.select(
        faces=lambda face: face.kind == "plane" and abs(face.normal[0]) < 1e-9
    ),
)
pipe.save("finned_pipe_heat_transfer.step")
