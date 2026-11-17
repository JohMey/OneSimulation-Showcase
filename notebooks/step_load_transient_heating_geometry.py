"""Geometry of the OneSimulation example "Transient heating with a step change in the load".

Builds the silicon disc, 20 mm in diameter and 4 mm thick, with the CAD objects of OneSimulation
and writes step_load_transient_heating.step. The heated face keeps its name top; the bottom and
the rim form the group insulated.
"""

from onesim.cad.d3 import Cylinder

disc = Cylinder(radius=0.010, length=0.004)
faces = disc.boundaries
disc = disc.grouped(insulated=faces.bottom | faces.outer_wall)
disc.save("step_load_transient_heating.step")
