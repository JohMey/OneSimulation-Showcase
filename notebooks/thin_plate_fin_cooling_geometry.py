"""Geometry of the OneSimulation example "Thin plate cooled on both faces".

Builds the aluminium plate, 100 mm x 100 mm and 1 mm thick, with the CAD objects of OneSimulation and
writes thin_plate_fin_cooling.step. The held edge is named base, the two large faces form the group
faces (cooled by the liquid) and the three other edges form the group edges (insulated).
"""

from onesim.cad.d3 import Box

plate = Box(length=0.1, width=0.1, height=0.001)
f = plate.boundaries
plate = plate.grouped(
    base=f.x_min, faces=f.z_min | f.z_max, edges=f.x_max | f.y_min | f.y_max
)
plate.save("thin_plate_fin_cooling.step")
