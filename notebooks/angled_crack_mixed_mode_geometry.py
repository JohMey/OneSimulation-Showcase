"""Geometry of the OneSimulation example "Mixed-mode stress intensity of an angled crack".

Builds a square steel plate, 200 mm wide and high and 10 mm thick, with a central through crack
of length 2a = 20 mm inclined at the angle beta to the x axis, with the CAD objects of
OneSimulation, once for each beta = 0, 30, 45 and 60 degrees. The crack is a slender diamond cut,
0.1 mm wide at its middle, with sharp tips. Each part is written to
angled_crack_mixed_mode_<beta>.step; the faces keep the names x_min, x_max, y_min, y_max, the crack
flanks are the face crack.side.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

side, thickness, half_length, gap = 0.200, 0.010, 0.010, 0.0001
for degrees in (0, 30, 45, 60):
    beta = np.radians(degrees)
    along, across = np.array([np.cos(beta), np.sin(beta)]), np.array(
        [-np.sin(beta), np.cos(beta)]
    )
    corners = [
        side / 2 + offset
        for offset in (
            -half_length * along,
            gap / 2 * across,
            half_length * along,
            -gap / 2 * across,
        )
    ]
    crack = Prism(
        profile=Profile.polygon([tuple(point) for point in corners]), height=thickness
    )
    (Box(length=side, width=side, height=thickness) - crack.named("crack")).save(
        f"angled_crack_mixed_mode_{degrees}.step"
    )
