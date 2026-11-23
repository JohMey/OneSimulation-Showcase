"""Geometry of the OneSimulation example "Laser-flash method on a coin-shaped sample".

Builds the ceramic disc, 12.7 mm in diameter and 2 mm thick, with the CAD objects of OneSimulation
and writes flash_method_axisymmetric.step. The flashed face is named front, the observed face rear
and the cylindrical face rim.
"""

from onesim.cad.d3 import Cylinder

disc = Cylinder(radius=6.35e-3, length=2.0e-3)
faces = disc.boundaries
disc = disc.grouped(front=faces.bottom, rear=faces.top, rim=faces.outer_wall)
disc.save("flash_method_axisymmetric.step")
