"""Geometry of the OneSimulation example "User-defined polynomial hyperelastic rubber".

Builds the rubber test block, a cube of 20 mm side, with the CAD objects of OneSimulation and
writes polynomial_hyperelastic_block.step. The faces keep the names x_min, x_max, y_min, y_max,
z_min and z_max.
"""

from onesim.cad.d3 import Box

Box(length=0.02, width=0.02, height=0.02).save("polynomial_hyperelastic_block.step")
