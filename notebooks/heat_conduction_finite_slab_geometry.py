"""Geometry of the OneSimulation example "Heat conduction in a finite slab".

Builds the slab, 100 mm thick along x with a 20 mm x 20 mm cross-section, with the CAD objects of
OneSimulation and writes heat_conduction_finite_slab.step. The two surfaces keep their names x_min
and x_max in the file; the four lateral faces form the group sides.
"""

from onesim.cad.d3 import Box

slab = Box(length=0.100, width=0.020, height=0.020)
faces = slab.boundaries
slab = slab.grouped(sides=faces.y_min | faces.y_max | faces.z_min | faces.z_max)
slab.save("heat_conduction_finite_slab.step")
