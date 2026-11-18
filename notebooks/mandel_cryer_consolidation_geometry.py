"""Geometry of the OneSimulation example "Mandel-Cryer effect in a consolidating sphere".

Builds one octant of the soil sphere (radius 1 m) with the CAD objects of OneSimulation and writes
mandel_cryer_consolidation.step. The file names the faces surface, plane_x, plane_y and plane_z.
"""

from onesim.cad.d3 import Box, Sphere

octant = Sphere(radius=1.0) & Box(length=1.0, width=1.0, height=1.0)
octant = octant.grouped(
    surface=octant.select(faces=lambda face: face.kind == "sphere"),
    plane_x=octant.select(faces=lambda face: face.kind == "plane" and abs(face.center[0]) < 1e-9),
    plane_y=octant.select(faces=lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-9),
    plane_z=octant.select(faces=lambda face: face.kind == "plane" and abs(face.center[2]) < 1e-9),
)
octant.save("mandel_cryer_consolidation.step")
