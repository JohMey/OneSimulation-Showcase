"""Geometry of the OneSimulation example "Tapered cantilever with two load cases".

Builds the tapered cantilever, 300 mm long, 60 mm deep at the clamped root and 30 mm deep at the
free tip, with both long edges inclined symmetrically, as a 5 mm thick plate with the CAD objects
of OneSimulation, and writes tapered_cantilever_load_cases.step. The faces are named root, tip,
top and bottom.
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Prism

length, root, tip, thickness = 0.300, 0.060, 0.030, 0.005
outline = Profile.polygon(
    [(0.0, -root / 2), (length, -tip / 2), (length, tip / 2), (0.0, root / 2)],
    names=("bottom", "tip", "top", "root"),
)
beam = Prism(profile=outline, height=thickness)
beam.save("tapered_cantilever_load_cases.step")
