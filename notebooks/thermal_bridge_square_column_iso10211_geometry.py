"""Geometry of the OneSimulation example "Thermal bridge: half of a square column (EN ISO 10211 case 1)".

Builds half of the square column, 1 m wide (x), 2 m high (y) and 0.1 m deep, with the CAD objects of
OneSimulation and writes thermal_bridge_square_column_iso10211.step. The warm top face is named warm,
the external face and the bottom form the group cold, and the cut through the middle of the column is
named symmetry. The example computes on the cross-section z = 0.05 m.
"""

from onesim.cad.d3 import Box

column = Box(length=1.0, width=2.0, height=0.1)
faces = column.boundaries
column = column.grouped(
    warm=faces.y_max, cold=faces.x_min | faces.y_min, symmetry=faces.x_max
)
column.save("thermal_bridge_square_column_iso10211.step")
