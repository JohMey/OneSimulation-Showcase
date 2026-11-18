"""Geometry of the OneSimulation example "Lid-driven cavity at Re = 100".

Builds the cavity, a square of side 1 m as a slab 62.5 mm thick, with the CAD objects of
OneSimulation and writes lid_driven_cavity.step. The lid is the face y_max; the faces keep their
names x_min, x_max, y_min, y_max, z_min and z_max in the file. The example computes on the
cross-section z = 31.25 mm.
"""

from onesim.cad.d3 import Box

cavity = Box(length=1.0, width=1.0, height=0.0625)
cavity.save("lid_driven_cavity.step")
