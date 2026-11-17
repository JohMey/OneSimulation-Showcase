"""Geometry of the OneSimulation example "Steady advection-diffusion in a channel".

Builds the channel, 100 mm long with a 10 mm x 10 mm cross-section, with the CAD objects of
OneSimulation and writes advection_diffusion_1d.step. Inlet and outlet keep their names x_min and
x_max in the file; the four lateral walls form the group walls.
"""

from onesim.cad.d3 import Box

channel = Box(length=0.100, width=0.010, height=0.010)
faces = channel.boundaries
channel = channel.grouped(walls=faces.y_min | faces.y_max | faces.z_min | faces.z_max)
channel.save("advection_diffusion_1d.step")
