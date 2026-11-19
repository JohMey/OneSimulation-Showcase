"""Geometry of the OneSimulation example "TE11 cutoff of a circular waveguide".

Builds the air volume of the guide, radius 20 mm and 60 mm long, with the CAD objects of
OneSimulation and writes circular_waveguide_te11_cutoff.step. The file names three face groups:
`port` (the feed disk at z = 0), `short` (the far end) and `walls` (the cylindrical wall).
"""

from onesim.cad.d3 import Cylinder

guide = Cylinder(radius=0.020, length=0.060)
faces = guide.boundaries
guide = guide.grouped(
    port=faces.bottom, short=faces.top, walls=faces.all - faces.bottom - faces.top
)
guide.save("circular_waveguide_te11_cutoff.step")
