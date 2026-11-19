"""Geometry of the OneSimulation example "Differentially heated square cavity".

Builds the cavity, a square of side 1 (dimensionless units) as a slab 0.1 thick, with the CAD
objects of OneSimulation and writes differentially_heated_square_cavity.step. The faces are named
hot (x = 0), cold (x = 1) and insulated (bottom and top); the example computes on the
cross-section z = 0.05.
"""

from onesim.cad.d3 import Box

cavity = Box(length=1.0, width=1.0, height=0.1)
cavity = cavity.grouped(
    hot=cavity.boundaries.x_min,
    cold=cavity.boundaries.x_max,
    insulated=cavity.boundaries.y_min | cavity.boundaries.y_max,
)
cavity.save("differentially_heated_square_cavity.step")
