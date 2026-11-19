"""Geometry of the OneSimulation example "Seawater intrusion in a coastal aquifer".

Builds the vertical section of the confined coastal aquifer of Henry's problem, 2 m long and
1 m deep, as a thin block with the CAD objects of OneSimulation and writes
seawater_intrusion_henry.step; the model lives on its section. The faces keep the names x_min
(inland boundary with the freshwater inflow), x_max (sea side), y_min (base) and y_max (top).
"""

from onesim.cad.d3 import Box

Box(length=2.0, width=1.0, height=0.1).save("seawater_intrusion_henry.step")
