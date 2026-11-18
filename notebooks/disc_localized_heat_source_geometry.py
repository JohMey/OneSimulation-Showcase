"""Geometry of the OneSimulation example "Disc heated by a localized source".

Builds the steel disc, 100 mm in radius and 2 mm thick, with the CAD objects of OneSimulation and
writes disc_localized_heat_source.step. The rim, held at a fixed temperature, is named rim; the two
flat faces form the group faces. The example computes on the mid-plane section of the disc.
"""

from onesim.cad.d3 import Cylinder

disc = Cylinder(radius=0.1, length=0.002)
faces = disc.boundaries
disc = disc.grouped(rim=faces.outer_wall, faces=faces.top | faces.bottom)
disc.save("disc_localized_heat_source.step")
