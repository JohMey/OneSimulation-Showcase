"""Geometry of the OneSimulation example "Primary creep under a changing load".

Builds a small square block of a creep specimen, 10 mm x 10 mm x 2 mm, with the CAD objects of
OneSimulation and writes primary_creep_nonconstant_load.step. The faces keep their names x_min,
x_max, y_min, y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

block = Box(length=0.010, width=0.010, height=0.002)
block.save("primary_creep_nonconstant_load.step")
