"""Geometry of the OneSimulation example "Rising bubble benchmark".

Builds the liquid column of the rising-bubble benchmark, 1 m wide and 2 m high, as a slab 0.5 m deep
with the CAD objects of OneSimulation and writes rising_bubble_benchmark.step. The faces are named
floor, lid and sides; the example computes on the cross-section at half depth.
"""

from onesim.cad.d3 import Box

column = Box(length=1.0, width=2.0, height=0.5)
column = column.grouped(floor=column.boundaries.y_min, lid=column.boundaries.y_max,
                        sides=column.boundaries.x_min | column.boundaries.x_max)
column.save("rising_bubble_benchmark.step")
