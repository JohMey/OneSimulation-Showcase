"""Geometry of the OneSimulation example "On-axis field of a solenoid".

Builds the air cylinder around the coil, radius 120 mm and 300 mm long, with the CAD objects of
OneSimulation and writes solenoid_on_axis_field.step. The faces keep their names outer_wall, bottom
and top in the file. The winding is no part of the file: the notebook declares it as a current
source, and the mesher embeds its volume in the air.
"""

from onesim.cad.d3 import Cylinder

air = Cylinder(radius=0.120, length=0.300)
air.save("solenoid_on_axis_field.step")
