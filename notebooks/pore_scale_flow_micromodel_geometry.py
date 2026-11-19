"""Geometry of the OneSimulation example "Pore-scale flow through a micromodel".

Builds a micromodel with the CAD objects of OneSimulation: a channel 8 mm long and 3 mm wide whose
middle 6 mm hold 32 cylindrical grains of radius 0.15 to 0.35 mm, placed at random (fixed seed)
with gaps of at least 0.12 mm between grains and to the side walls. The part is a slab 0.2 mm thick
written to pore_scale_flow_micromodel.step; the faces keep their names x_min (inlet), x_max
(outlet), y_min and y_max (side walls), the grain surfaces are grain.outer_wall. The example
computes on the cross-section z = 0.1 mm.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

mm = 1e-3
length, width, thickness, gap = 8 * mm, 3 * mm, 0.2 * mm, 0.12 * mm
rng = np.random.default_rng(7)
grains = []
while len(grains) < 32:
    radius = rng.uniform(0.15, 0.35) * mm
    centre = rng.uniform(
        (1 * mm + radius, radius + gap), (7 * mm - radius, width - radius - gap)
    )
    if all(
        np.hypot(*(centre - other)) > radius + other_radius + gap
        for other, other_radius in grains
    ):
        grains.append((centre, radius))

micromodel = Box(length=length, width=width, height=thickness)
for centre, radius in grains:
    micromodel = micromodel - Cylinder(
        radius=radius, length=3 * thickness, origin=(centre[0], centre[1], -thickness)
    ).named("grain")
micromodel.save("pore_scale_flow_micromodel.step")
