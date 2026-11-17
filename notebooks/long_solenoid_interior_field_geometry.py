"""Geometry of the OneSimulation example "Interior field of a long solenoid".

Builds the air cylinder around the coil, radius 50 mm and 400 mm long, with the CAD objects of
OneSimulation and writes long_solenoid_interior_field.step. The faces keep their names outer_wall,
bottom and top in the file. The winding is no part of the file: the notebook declares it as a
current source, and the mesher embeds its volume in the air.
"""

from onesim.cad.d3 import Cylinder

air = Cylinder(radius=0.050, length=0.400)
air.save("long_solenoid_interior_field.step")
