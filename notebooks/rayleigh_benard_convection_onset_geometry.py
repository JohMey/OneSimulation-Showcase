"""Geometry of the OneSimulation example "Rayleigh-Benard convection: onset of the rolls".

Builds a horizontal fluid layer of depth 1 (dimensionless units) and width 2*pi/3.117, one
critical wavelength, as a slab 0.1 thick with the CAD objects of OneSimulation and writes
rayleigh_benard_convection_onset.step. The faces are named heated_bottom, cooled_top and sides; the
example computes on the cross-section z = 0.05.
"""

import numpy as np
from onesim.cad.d3 import Box

layer = Box(length=2 * np.pi / 3.117, width=1.0, height=0.1)
layer = layer.grouped(
    heated_bottom=layer.boundaries.y_min,
    cooled_top=layer.boundaries.y_max,
    sides=layer.boundaries.x_min | layer.boundaries.x_max,
)
layer.save("rayleigh_benard_convection_onset.step")
