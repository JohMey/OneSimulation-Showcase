"""Geometry of the OneSimulation example "Counterflow heat exchanger".

Builds one channel of the exchanger, 45 mm long with a 1 mm gap (along y) and 2 mm width (along z),
with the CAD objects of OneSimulation and writes counterflow_heat_exchanger_ntu.step. Both streams
use this channel. The faces keep their names x_min, x_max, y_min, y_max, z_min and z_max in the
file; the two faces across the width form the group ends.
"""

from onesim.cad.d3 import Box

duct = Box(length=0.045, width=1.0e-3, height=2.0e-3)
faces = duct.boundaries
duct = duct.grouped(ends=faces.z_min | faces.z_max)
duct.save("counterflow_heat_exchanger_ntu.step")
