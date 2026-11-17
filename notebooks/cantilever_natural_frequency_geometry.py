"""Geometry of the OneSimulation example "Natural frequency of a clamped beam".

Builds the beam, 120 mm long with a square 12 mm cross-section, with the CAD objects of
OneSimulation and writes cantilever_natural_frequency.step. The faces keep their names x_min,
x_max, y_min, y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

beam = Box(length=0.120, width=0.012, height=0.012)
beam.save("cantilever_natural_frequency.step")
