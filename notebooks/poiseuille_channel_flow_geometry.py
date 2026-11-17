"""Geometry of the OneSimulation example "Poiseuille flow between parallel plates".

Builds a short piece of the channel, 20 mm long, with a 10 mm gap between the plates and 5 mm wide,
with the CAD objects of OneSimulation and writes poiseuille_channel_flow.step. The plates are the
faces y_min and y_max; the faces keep their names x_min, x_max, y_min, y_max, z_min and z_max in
the file.
"""

from onesim.cad.d3 import Box

channel = Box(length=0.020, width=0.010, height=0.005)
channel.save("poiseuille_channel_flow.step")
