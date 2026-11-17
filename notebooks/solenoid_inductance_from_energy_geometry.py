"""Geometry of the OneSimulation example "Inductance from the magnetic field energy".

Builds the air cylinder around the coil, radius 100 mm and 360 mm long, with the CAD objects of
OneSimulation and writes solenoid_inductance_from_energy.step. The faces keep their names
outer_wall, bottom and top in the file. The winding is no part of the file: the notebook declares it
as a current source, and the mesher embeds its volume in the air.
"""

from onesim.cad.d3 import Cylinder

air = Cylinder(radius=0.100, length=0.360)
air.save("solenoid_inductance_from_energy.step")
