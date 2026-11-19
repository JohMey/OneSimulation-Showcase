"""Geometry of the OneSimulation example "Supersonic flow over a bump in a channel (3D)".

Builds half of a channel 1 m long, 0.4 m high and 0.3 m wide with the CAD objects of
OneSimulation: the half 0 < z < 0.15 m next to the mid-plane z = 0. On the floor sits a bump that
covers the middle half of the channel width (0 < z < 0.075 m in the half model): it rises at 10
degrees from x = 0.4 m, stays level from x = 0.48 m to 0.52 m and falls at 10 degrees to the floor
at x = 0.6 m. The gas region is written to supersonic_bump_channel_3d.step; the channel faces keep
the names x_min (inlet), x_max (outlet), y_min (floor), y_max (ceiling), z_min (mid-plane) and
z_max (side wall), the faces of the bump are named bump.*.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

rise = 0.08 * np.tan(np.radians(10))
bump = Prism(
    profile=Profile.polygon(
        [(0.4, -0.01), (0.4, 0.0), (0.48, rise), (0.52, rise), (0.6, 0.0), (0.6, -0.01)]
    ),
    height=0.085,
    origin=(0.0, 0.0, -0.01),
)
channel = Box(length=1.0, width=0.4, height=0.15) - bump.named("bump")
channel.save("supersonic_bump_channel_3d.step")
