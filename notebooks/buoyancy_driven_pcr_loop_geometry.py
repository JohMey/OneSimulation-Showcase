"""Geometry of the OneSimulation example "Buoyancy-driven PCR in a triangular loop".

Builds the closed triangular loop channel of a convective PCR chip with the CAD objects of
OneSimulation: an equilateral triangle standing upright in the x-y plane (gravity along -y), 20 mm
side length along the centre line of the channel, with a square channel cross-section of 1 x 1 mm.
The loop is symmetric about its mid-depth plane; the file holds the front half (0.5 mm deep).
Writes buoyancy_driven_pcr_loop.step with the face groups denaturation (both walls of the left
leg), annealing (right leg), extension (bottom leg), front (the front cover) and mid_plane.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Prism

side, channel = 20e-3, 1e-3
corners = lambda inradius: [
    (-np.sqrt(3) * inradius, -inradius),
    (np.sqrt(3) * inradius, -inradius),
    (0.0, 2 * inradius),
]
centre = side / (2 * np.sqrt(3))
outer = Profile.polygon(
    corners(centre + channel / 2), names=("extension", "annealing", "denaturation")
)
inner = Profile.polygon(
    corners(centre - channel / 2), names=("extension", "annealing", "denaturation")
)
loop = Prism(profile=outer.with_holes(inner), height=channel / 2)
loop = loop.grouped(front=loop.boundaries.bottom, mid_plane=loop.boundaries.top)
loop.save("buoyancy_driven_pcr_loop.step")
