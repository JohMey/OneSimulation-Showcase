"""Geometry of the OneSimulation example "Melting of a phase change material in a heated corner".

Builds the block of paraffin, 60 mm x 60 mm in cross-section and 10 mm deep, with the CAD objects of
OneSimulation and writes pcm_corner_melting.step. The two heated walls form the group heated, the two
other walls the group insulated. The example computes on the cross-section z = 5 mm.
"""

from onesim.cad.d3 import Box

block = Box(length=0.06, width=0.06, height=0.01)
faces = block.boundaries
block = block.grouped(
    heated=faces.x_min | faces.y_min, insulated=faces.x_max | faces.y_max
)
block.save("pcm_corner_melting.step")
