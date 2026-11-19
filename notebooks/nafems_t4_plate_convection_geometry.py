"""Geometry of the OneSimulation example "Two-dimensional conduction with convection (NAFEMS T4)".

Builds the plate, 0.6 m wide, 1.0 m high and 0.1 m deep, with the CAD objects of OneSimulation and
writes nafems_t4_plate_convection.step. The heated base is named base, the insulated left side
insulated and the two cooled sides form the group cooled. The example computes on the
cross-section z = 0.05 m.
"""

from onesim.cad.d3 import Box

plate = Box(length=0.6, width=1.0, height=0.1)
faces = plate.boundaries
plate = plate.grouped(
    base=faces.y_min, insulated=faces.x_min, cooled=faces.x_max | faces.y_max
)
plate.save("nafems_t4_plate_convection.step")
