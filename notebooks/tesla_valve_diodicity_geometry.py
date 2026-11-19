"""Geometry of the OneSimulation example "Tesla microvalve: diodicity over the Reynolds number".

Builds a single-stage Tesla valve with the CAD objects of OneSimulation: a straight channel 1 mm
wide and 26 mm long, and a loop channel of the same width whose two legs leave the main channel at
30 degrees, leaning against the forward flow, joined by a rounded top 4 mm above the channel floor.
The part is a slab 0.5 mm thick written to tesla_valve_diodicity.step; the channel ends are the faces
channel.x_min (forward inlet) and channel.x_max. The example computes on the cross-section
z = 0.25 mm.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

mm = 1e-3
width, thickness, run = 1.0 * mm, 0.5 * mm, 1 / np.tan(np.radians(30))
outer = Profile.rounded_polygon([(7 * mm, 0.0), (17 * mm, 0.0), ((17 - 4 * run) * mm, 4 * mm), ((7 - 4 * run) * mm, 4 * mm)], 1.3 * mm)
island = Profile.rounded_polygon([((9 - run) * mm, width), ((15 - run) * mm, width), ((15 - 3 * run) * mm, 3 * mm),
                                  ((9 - 3 * run) * mm, 3 * mm)], 0.3 * mm)
channel = Box(length=26 * mm, width=width, height=thickness, origin=(-4 * mm, 0.0, 0.0)).named("channel")
valve = (channel | Prism(profile=outer, height=thickness)) - Prism(profile=island, height=thickness)
valve.save("tesla_valve_diodicity.step")
