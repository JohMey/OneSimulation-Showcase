"""Geometry of the OneSimulation example "Ear canal acoustics".

Builds the air of an ear canal with the CAD objects of OneSimulation: a circle of radius 3.75 mm
swept along a bent centre line - 13.5 mm straight, a 60 degree bend of radius 10 mm, then straight
on to the eardrum - 25 mm long in total. It writes ear_canal_transfer.step with the faces
entrance, eardrum and wall.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Sweep

radius, bend, first, second = 0.00375, np.radians(60.0), 0.0135, 0.012575
corner = (0.0, 0.0, first)
end = (second * np.sin(bend), 0.0, first + second * np.cos(bend))
circle = Profile(
    Path(
        (radius, 0.0),
        (
            Segment((-radius, 0.0), center=(0.0, 0.0)),
            Segment((radius, 0.0), center=(0.0, 0.0)),
        ),
    )
)
canal = Sweep(profile=circle, path=((0.0, 0.0, 0.0), corner, end), corner_radius=0.010)
faces = canal.boundaries
canal.grouped(entrance=faces.start, eardrum=faces.end, wall=faces.side).save(
    "ear_canal_transfer.step"
)
