"""Geometry of the OneSimulation example "Laser heating of a rotating silicon wafer".

Builds the silicon wafer, 100 mm in diameter and 0.5 mm thick, with the CAD objects of OneSimulation
and writes laser_heating_rotating_wafer.step. The face under the laser is named top; the bottom and
the rim form the group insulated.
"""

from onesim.cad.d3 import Cylinder

wafer = Cylinder(radius=0.05, length=0.0005)
f = wafer.boundaries
wafer = wafer.grouped(insulated=f.bottom | f.outer_wall)
wafer.save("laser_heating_rotating_wafer.step")
