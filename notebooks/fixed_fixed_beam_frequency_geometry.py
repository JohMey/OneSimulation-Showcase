"""Geometry of the OneSimulation example "Natural frequency of a beam clamped at both ends".

Builds the beam, 240 mm long with a square 12 mm cross-section, with the CAD objects of
OneSimulation and writes fixed_fixed_beam_frequency.step. The faces keep their names x_min,
x_max, y_min, y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

beam = Box(length=0.240, width=0.012, height=0.012)
beam.save("fixed_fixed_beam_frequency.step")
