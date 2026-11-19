"""Geometry of the OneSimulation example "Inflation of an artery wall with collagen fibres".

Builds a quarter of a tube segment of the arterial media with the CAD objects of OneSimulation:
inner radius 0.71 mm, wall thickness 0.26 mm, length 0.5 mm. The part is written to
arterial_wall_inflation.step with the faces lumen (inner wall), adventitia (outer wall),
plane_x (x = 0), plane_y (y = 0), bottom (z = 0) and top (z = length).
"""

from onesim.cad.d3 import AnnularCylinder, Box

inner, wall, length = 0.71e-3, 0.26e-3, 0.5e-3
tube = AnnularCylinder(
    inner_radius=inner, outer_radius=inner + wall, length=length
) & Box(length=2e-3, width=2e-3, height=2 * length, origin=(0.0, 0.0, -length / 2))
tube = tube.grouped(
    lumen=tube.select(
        faces=lambda face: face.kind == "cylinder"
        and abs(face.center[0] ** 2 + face.center[1] ** 2) < (inner + wall / 2) ** 2
    ),
    adventitia=tube.select(
        faces=lambda face: face.kind == "cylinder"
        and abs(face.center[0] ** 2 + face.center[1] ** 2) > (inner + wall / 2) ** 2
    ),
    plane_x=tube.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0]) < 1e-12
    ),
    plane_y=tube.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-12
    ),
    bottom=tube.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[2]) < 1e-12
    ),
    top=tube.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[2] - length) < 1e-12
    ),
)
tube.save("arterial_wall_inflation.step")
