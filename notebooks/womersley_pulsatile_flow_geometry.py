"""Geometry of the OneSimulation example "Womersley pulsatile flow between parallel plates".

Builds a thin slab of the channel, 0.5 mm long and wide, with a gap of 2 mm between the plates,
with the CAD objects of OneSimulation and writes womersley_pulsatile_flow.step. The plates are the
faces y_min and y_max; the faces keep their names x_min, x_max, y_min, y_max, z_min and z_max in
the file.
"""

from onesim.cad.d3 import Box

channel = Box(length=0.0005, width=0.002, height=0.0005)
channel.save("womersley_pulsatile_flow.step")
