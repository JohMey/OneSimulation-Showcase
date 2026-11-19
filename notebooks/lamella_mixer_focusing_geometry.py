"""Geometry of the OneSimulation example "Lamella mixer with hydrodynamic focusing".

Builds the lower half of a lamella micromixer with the CAD objects of OneSimulation: an inlet
chamber 4 mm wide and 2 mm long, a contraction over 2 mm to the mixing channel 1 mm wide and 12 mm
long; the channel is 0.5 mm deep, the file holds the 0.25 mm below the mid-depth plane. The part is
written to lamella_mixer_focusing.step with the faces inlet, outlet, wall (side walls), bottom and top
(the mid-depth plane).
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Prism

mm = 1e-3
outline = [(0, -2), (2, -2), (4, -0.5), (16, -0.5), (16, 0.5), (4, 0.5), (2, 2), (0, 2)]
sides = ["wall", "wall", "wall", "outlet", "wall", "wall", "wall", "inlet"]
mixer = Prism(
    profile=Profile.polygon([(x * mm, y * mm) for x, y in outline], names=sides),
    height=0.25 * mm,
)
mixer.save("lamella_mixer_focusing.step")
