"""Geometry of the OneSimulation example "Mode shapes of a clamped-free beam".

Builds the beam, 120 mm long with a 12 mm x 8 mm cross-section, with the CAD objects of
OneSimulation and writes beam_mode_shapes.step. The faces keep their names x_min, x_max, y_min,
y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

beam = Box(length=0.120, width=0.012, height=0.008)
beam.save("beam_mode_shapes.step")
