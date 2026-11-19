"""Geometry of the OneSimulation example "Temperature field in a cooling flange".

Builds one half (z >= 0) of the glass flange with the CAD objects of OneSimulation and writes
cooling_flange_temperature.step: a pipe of 16 mm bore and 4 mm wall with three fins, 4 mm wide and
10 mm tall, with rounded shoulders and tips, joined to a 12 mm connecting pipe with a 3 mm wall.
Both pieces are solids of revolution about the pipe axis x. The file names the face groups bore,
outer, pipe_end, flange_end and symmetry.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution, UnionFromObjects

mm = 1e-3
bore, pipe_length, pipe_wall, wall, fin_height, fin_width = (
    8 * mm,
    12 * mm,
    3 * mm,
    4 * mm,
    10 * mm,
    4 * mm,
)
root, shoulder, top = (
    bore + wall,
    bore + wall + fin_width / 2,
    bore + wall + fin_height - fin_width / 2,
)

connecting_pipe = Path(
    (bore, -pipe_length),
    (
        Segment((bore + pipe_wall, -pipe_length)),
        Segment((bore + pipe_wall, -2 * pipe_length / 3)),
        Segment(
            (root, 0.0), control=((root, -pipe_length / 3), (root, -pipe_length / 3))
        ),
        Segment((bore, 0.0)),
        Segment((bore, -pipe_length)),
    ),
)
fins = [Segment((root, 0.0))]
for fin in range(3):
    start, rise, fall, end = (
        2 * fin_width * fin + offset
        for offset in (0, fin_width / 2, 3 * fin_width / 2, 2 * fin_width)
    )
    fins += [
        Segment((shoulder, rise), control=((root, rise),)),
        Segment((top, rise)),
        Segment((top, fall), center=(top, start + fin_width)),
        Segment((shoulder, fall)),
        Segment((root, end), control=((root, fall),)),
    ]
fins += [Segment((bore, 6 * fin_width)), Segment((bore, 0.0))]

half_turn = dict(angle=np.pi, axis=(1.0, 0.0, 0.0))
flange = UnionFromObjects(
    objects=[
        Revolution(profile=Profile(connecting_pipe), **half_turn),
        Revolution(profile=Profile(Path((bore, 0.0), tuple(fins))), **half_turn),
    ]
)

end_face = (
    lambda face, x: face.kind == "plane"
    and abs(abs(face.normal[0]) - 1) < 1e-9
    and abs(face.center[0] - x) < 1e-9
)
pipe_end = flange.select(faces=lambda face: end_face(face, -pipe_length))
flange_end = flange.select(faces=lambda face: end_face(face, 6 * fin_width))
symmetry = flange.select(
    faces=lambda face: face.kind == "plane" and abs(abs(face.normal[2]) - 1) < 1e-9
)
inner = flange.select(
    faces=lambda face: face.kind == "cylinder" and abs(face.radius - bore) < 1e-9
)
flange = flange.grouped(
    pipe_end=pipe_end,
    flange_end=flange_end,
    symmetry=symmetry,
    bore=inner,
    outer=flange.boundaries.all - pipe_end - flange_end - symmetry - inner,
)
flange.save("cooling_flange_temperature.step")
