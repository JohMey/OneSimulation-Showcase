"""Geometry of the OneSimulation example "Effective diffusivity of a porous medium".

Builds, as thin plates with the CAD objects of OneSimulation, a strip of a porous medium: a
square array of 6 x 2 impermeable cylindrical grains in cells of 1 mm, for four solid fractions
(porous_effective_diffusivity.step for 50 %, porous_effective_diffusivity_<percent>.step otherwise). The faces x_min, x_max, y_min and y_max of the strip
keep their names, the grain surfaces are grouped as "grains". The example computes on the
cross-section z = 0.05 mm.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

cell, thickness = 1e-3, 1e-4
for fraction in (0.1, 0.3, 0.5, 0.7):
    radius = cell * np.sqrt(fraction / np.pi)
    strip = Box(length=6 * cell, width=2 * cell, height=thickness)
    for i in range(6):
        for j in range(2):
            strip = strip - Cylinder(
                radius=radius, length=thickness, origin=((i + 0.5) * cell, (j + 0.5) * cell, 0.0)
            )
    strip = strip.grouped(grains=strip.select(faces=lambda face: face.kind == "cylinder"))
    filename = (
        "porous_effective_diffusivity.step"
        if fraction == 0.5
        else f"porous_effective_diffusivity_{round(100 * fraction)}.step"
    )
    strip.save(filename)
