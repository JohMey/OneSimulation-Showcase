"""Geometry of the OneSimulation example "Bending of a cantilever beam".

Builds the beam, 120 mm long with an 18 mm x 12 mm cross-section, with the CAD objects of
OneSimulation and writes cantilever_beam_bending.step. The faces keep their names x_min, x_max,
y_min, y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

beam = Box(length=0.120, width=0.018, height=0.012)
beam.save("cantilever_beam_bending.step")
