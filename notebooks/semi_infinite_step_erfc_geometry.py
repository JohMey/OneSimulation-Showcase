"""Geometry of the OneSimulation example "Surface temperature step on a long bar".

Builds the bar, 200 mm long with a 10 mm x 10 mm cross-section, with the CAD objects of
OneSimulation and writes semi_infinite_step_erfc.step. The end faces keep their names x_min and
x_max in the file; the four lateral faces form the group sides.
"""

from onesim.cad.d3 import Box

bar = Box(length=0.200, width=0.010, height=0.010)
faces = bar.boundaries
bar = bar.grouped(sides=faces.y_min | faces.y_max | faces.z_min | faces.z_max)
bar.save("semi_infinite_step_erfc.step")
