"""Geometry of the OneSimulation example "Spinodal decomposition benchmark".

Builds the benchmark square, 200 x 200 in the dimensionless units of the benchmark and 1 thick,
with the CAD objects of OneSimulation and writes spinodal_decomposition_benchmark.step. The faces
keep their names x_min, x_max, y_min, y_max, z_min and z_max in the file; the example computes on
the mid-plane.
"""

from onesim.cad.d3 import Box

square = Box(length=200.0, width=200.0, height=1.0)
square.save("spinodal_decomposition_benchmark.step")
