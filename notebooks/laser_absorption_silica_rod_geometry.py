"""Geometry of the OneSimulation example "Laser beam absorbed in a silica glass rod".

Builds the glass rod, 30 mm in diameter and 20 mm long, with the CAD objects of OneSimulation and
writes laser_absorption_silica_rod.step. The beam enters through the face entrance and leaves through
the face exit; the cylindrical face is named mantle.
"""

from onesim.cad.d3 import Cylinder

rod = Cylinder(radius=0.015, length=0.020)
f = rod.boundaries
rod = rod.grouped(entrance=f.bottom, exit=f.top, mantle=f.outer_wall)
rod.save("laser_absorption_silica_rod.step")
