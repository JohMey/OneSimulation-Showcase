"""Geometry of the OneSimulation example "Elastoplastic perforated plate in plane stress".

Builds a quarter of a perforated strip (full plate 200 mm wide and 360 mm long with a central hole of
100 mm diameter, 1 mm thick) with the CAD objects of OneSimulation, names the faces the simulation
refers to and writes holed_plate_elastoplastic_plane_stress.step. The example computes on its
mid-plane section.
"""

from onesim.cad.d3 import Box, Cylinder

quarter = Box(length=0.100, width=0.180, height=0.001) - Cylinder(
    radius=0.050, length=0.003, origin=(0.0, 0.0, -0.001)
)
quarter = quarter.grouped(
    hole=quarter.select(faces=lambda face: face.kind == "cylinder"),
    symmetry_x=quarter.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0]) < 1e-9
    ),
    symmetry_y=quarter.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-9
    ),
    loaded=quarter.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1] - 0.180) < 1e-9
    ),
    side=quarter.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0] - 0.100) < 1e-9
    ),
)
quarter.save("holed_plate_elastoplastic_plane_stress.step")
