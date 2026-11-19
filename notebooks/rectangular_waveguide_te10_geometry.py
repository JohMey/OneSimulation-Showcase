"""Geometry of the OneSimulation example "TE10 mode of a rectangular waveguide".

Builds the air volume of the guide, 100 mm wide, 50 mm high and 80 mm long, with the CAD objects of
OneSimulation and writes rectangular_waveguide_te10.step. The file names three face groups: `port`
(the feed cross-section at z = 0), `short` (the far end) and `walls` (the four side walls).
"""

from onesim.cad.d3 import Box

guide = Box(length=0.100, width=0.050, height=0.080)
faces = guide.boundaries
guide = guide.grouped(
    port=faces.z_min, short=faces.z_max, walls=faces.all - faces.z_min - faces.z_max
)
guide.save("rectangular_waveguide_te10.step")
