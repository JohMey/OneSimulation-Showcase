"""Geometry of the OneSimulation example "Free convection of air between two vertical plates".

Builds the air gap between two vertical plates, 20 mm apart and 20 mm high, as a slab 5 mm deep
with the CAD objects of OneSimulation and writes air_cavity_free_convection.step. The faces are
named hot_plate (x = 0), cold_plate (x = 20 mm), floor and ceiling; the example computes on the
cross-section at half depth.
"""

from onesim.cad.d3 import Box

gap = Box(length=0.02, width=0.02, height=0.005)
gap = gap.grouped(
    hot_plate=gap.boundaries.x_min,
    cold_plate=gap.boundaries.x_max,
    floor=gap.boundaries.y_min,
    ceiling=gap.boundaries.y_max,
)
gap.save("air_cavity_free_convection.step")
