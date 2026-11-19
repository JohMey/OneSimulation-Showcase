"""Geometry of the OneSimulation example "Polarized ports of a circular waveguide".

Builds the air volume of a circular waveguide, 20 mm in radius and 120 mm long, with the CAD
objects of OneSimulation and writes polarized_circular_waveguide_port.step. The file names three
face groups: `port` (the feed cross-section at z = 0), `exit` (the far cross-section, terminated
by a matched boundary) and `wall` (the cylindrical metal wall).
"""

from onesim.cad.d3 import Cylinder

guide = Cylinder(radius=0.020, length=0.120)
faces = guide.boundaries
guide = guide.grouped(
    port=faces.bottom, exit=faces.top, wall=faces.all - faces.bottom - faces.top
)
guide.save("polarized_circular_waveguide_port.step")
