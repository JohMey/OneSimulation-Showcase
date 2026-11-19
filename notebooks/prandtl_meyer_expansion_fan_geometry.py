"""Geometry of the OneSimulation example "Supersonic expansion fan at a convex corner".

Builds the gas region above a wall that turns away from the stream by 10 degrees with the CAD
objects of OneSimulation: a straight wall 0.5 m long, then the turned wall to x = 1.5 m, an upper
boundary at y = 1 m; as a slab 0.1 m thick written to prandtl_meyer_expansion_fan.step. The side
faces are named inlet, wall (before the corner), ramp (after the corner), outlet and upper; the
example computes on the cross-section z = 0.05 m.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Prism

turn = np.radians(10)
outline = [(0.0, 0.0), (0.5, 0.0), (1.5, -np.tan(turn)), (1.5, 1.0), (0.0, 1.0)]
region = Prism(
    profile=Profile.polygon(
        outline, names=["wall", "ramp", "outlet", "upper", "inlet"]
    ),
    height=0.1,
)
region.save("prandtl_meyer_expansion_fan.step")
