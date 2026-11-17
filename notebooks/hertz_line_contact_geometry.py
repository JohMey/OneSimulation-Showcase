"""Geometry of the OneSimulation example "Hertz line contact".

Builds the elastic block, 6.4 wide, 3.6 deep and 1 thick (dimensionless units), with the CAD
objects of OneSimulation and writes hertz_line_contact.step. The faces keep their names x_min,
x_max, y_min, y_max, z_min and z_max in the file; the example computes on the cross-section z = 0.5.
"""

from onesim.cad.d3 import Box

block = Box(length=6.4, width=3.6, height=1.0)
block.save("hertz_line_contact.step")
