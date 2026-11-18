"""Geometry of the OneSimulation example "Uniaxial compression of a prestressed rock block".

Builds the cubic block of side 1 m with the CAD objects of OneSimulation and writes
prestressed_block_mohr_coulomb.step. The faces keep the names x_min, x_max, y_min, y_max, z_min
(base) and z_max (loaded top).
"""

from onesim.cad.d3 import Box

Box(length=1.0, width=1.0, height=1.0).save("prestressed_block_mohr_coulomb.step")
