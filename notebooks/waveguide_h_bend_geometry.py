"""Geometry of the OneSimulation example "H-plane bend of a rectangular waveguide".

Builds with the CAD objects of OneSimulation the air inside a WR-90 rectangular waveguide
(22.86 mm x 10.16 mm) that turns by 90 degrees in the plane of its broad walls: 30 mm of straight
guide along y, a circular bend with an inner wall radius of 10 mm, and 30 mm of straight guide along x.
The profile lies in the x-y plane and is extruded by the narrow side along z. Faces: port_in (y = 0),
port_out (the end of the x leg) and metal (all walls), in waveguide_h_bend.step.
"""

from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Prism

a, b, inner, straight = 0.02286, 0.01016, 0.010, 0.030
centre = (a + inner, straight)
outline = Path(
    (0.0, 0.0),
    (
        Segment((a, 0.0), name="port_in"),
        Segment((a, straight), name="wall"),
        Segment(
            (a + inner, straight + inner),
            center=centre,
            counterclockwise=False,
            name="wall",
        ),
        Segment((a + inner + straight, straight + inner), name="wall"),
        Segment((a + inner + straight, straight + inner + a), name="port_out"),
        Segment((a + inner, straight + inner + a), name="wall"),
        Segment((0.0, straight), center=centre, counterclockwise=True, name="wall"),
        Segment((0.0, 0.0), name="wall"),
    ),
)
guide = Prism(profile=Profile(outline), height=b)
faces = guide.boundaries
guide.grouped(metal=faces.wall | faces.top | faces.bottom).save("waveguide_h_bend.step")
