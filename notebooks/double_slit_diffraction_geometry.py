"""Geometry of the OneSimulation example "Double-slit diffraction of sound".

Builds the air of the double-slit experiment as a thin plate with the CAD objects of OneSimulation:
two waveguides 50 mm wide and 150 mm long, 300 mm apart, open through a rigid screen into a half
disc of air of radius 1 m. It writes double_slit_diffraction.step with the faces inlet, guide,
screen and outer.
"""

from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Prism

width, spacing, guide, outer = 0.05, 0.30, 0.15, 1.0
left, right = -spacing / 2, spacing / 2
outline = Path(
    (-outer, 0.0),
    (
        Segment((left - width / 2, 0.0), name="screen"),
        Segment((left - width / 2, -guide), name="guide"),
        Segment((left + width / 2, -guide), name="inlet"),
        Segment((left + width / 2, 0.0), name="guide"),
        Segment((right - width / 2, 0.0), name="screen"),
        Segment((right - width / 2, -guide), name="guide"),
        Segment((right + width / 2, -guide), name="inlet"),
        Segment((right + width / 2, 0.0), name="guide"),
        Segment((outer, 0.0), name="screen"),
        Segment((-outer, 0.0), center=(0.0, 0.0), name="outer"),
    ),
)
Prism(profile=Profile(outline), height=0.01).save("double_slit_diffraction.step")
