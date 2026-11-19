"""Geometry of the OneSimulation example "Bulk temperature in a heated square duct".

Builds one quarter of a square duct (2 x 2 mm cross-section, 30 mm long) with the CAD objects of
OneSimulation and writes bulk_temperature_square_duct.step. The flow runs along z; the faces keep
the names of the box: z_min (inlet), z_max (outlet), x_max and y_max (the heated walls) and x_min
and y_min (the symmetry planes through the duct axis).
"""

from onesim.cad.d3 import Box

Box(length=1.0e-3, width=1.0e-3, height=30.0e-3).save(
    "bulk_temperature_square_duct.step"
)
