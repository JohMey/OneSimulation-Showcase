"""Geometry of the OneSimulation example "Transmission loss of an expansion-chamber muffler".

Builds half of the air in a muffler with the CAD objects of OneSimulation: a cylindrical chamber
(radius 80 mm, length 300 mm) with an inlet pipe and an outlet pipe (radius 25 mm, length 100 mm),
offset by 40 mm to opposite sides of the axis. The muffler is cut in its plane of symmetry y = 0.
It writes expansion_chamber_muffler.step with the faces inlet, outlet, symmetry and walls.
"""

from onesim.cad.d3 import Box, Cylinder

chamber_radius, chamber_length, pipe_radius, pipe_length, offset = (
    0.08,
    0.30,
    0.025,
    0.10,
    0.04,
)
muffler = (
    Cylinder(radius=chamber_radius, length=chamber_length)
    | Cylinder(
        radius=pipe_radius, length=pipe_length, origin=(-offset, 0.0, -pipe_length)
    )
    | Cylinder(
        radius=pipe_radius, length=pipe_length, origin=(offset, 0.0, chamber_length)
    )
)
half = muffler - Box(length=0.4, width=0.2, height=0.8, origin=(-0.2, -0.2, -0.2))
plane = lambda z: (
    lambda face: face.kind == "plane"
    and abs(abs(face.normal[2]) - 1) < 1e-9
    and abs(face.center[2] - z) < 1e-9
)
inlet, outlet = half.select(faces=plane(-pipe_length)), half.select(
    faces=plane(chamber_length + pipe_length)
)
symmetry = half.select(
    faces=lambda face: face.kind == "plane" and abs(abs(face.normal[1]) - 1) < 1e-9
)
walls = half.select(
    faces=lambda face: not (
        face.kind == "plane"
        and (
            abs(abs(face.normal[1]) - 1) < 1e-9
            or abs(face.center[2] + pipe_length) < 1e-9
            or abs(face.center[2] - chamber_length - pipe_length) < 1e-9
        )
    )
)
half = half.grouped(inlet=inlet, outlet=outlet, symmetry=symmetry, walls=walls)
half.save("expansion_chamber_muffler.step")
