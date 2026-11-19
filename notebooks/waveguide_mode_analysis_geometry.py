"""Geometry of the OneSimulation example "Mode analysis of a rectangular waveguide cross-section".

Builds a short piece of the air volume of a WR-90 waveguide (inner cross-section 22.86 mm x
10.16 mm, 1.5 mm long) with the CAD objects of OneSimulation and writes
waveguide_mode_analysis.step. The file names three face groups: `port` (the cross-section at
z = 0 on which the modes are computed), `short` (the far end) and `walls` (the four side walls).
"""

from onesim.cad.d3 import Box

guide = Box(length=22.86e-3, width=10.16e-3, height=1.5e-3)
faces = guide.boundaries
guide = guide.grouped(
    port=faces.z_min, short=faces.z_max, walls=faces.all - faces.z_min - faces.z_max
)
guide.save("waveguide_mode_analysis.step")
