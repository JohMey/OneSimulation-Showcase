"""Geometry of the OneSimulation example "Stress around a small hole in a large plate (Kirsch)".

Builds one quarter of a square plate with a central circular hole of radius 10 mm, 10 mm thick,
with the CAD objects of OneSimulation. The reference plate has the half-width L = 20a and is
written to kirsch_plate_hole_stress.step; the plates with L = 5a, 10a and 40a for the plate-size
study go to kirsch_plate_hole_stress_5a.step, _10a.step and _40a.step. The faces are named hole,
loaded (x = L), symmetry_x (x = 0) and symmetry_y (y = 0).
"""

from onesim.cad.d3 import Box, Cylinder

radius, thickness = 0.010, 0.010
for size, suffix in ((20, ""), (5, "_5a"), (10, "_10a"), (40, "_40a")):
    plate = Box(length=size * radius, width=size * radius, height=thickness) - Cylinder(
        radius=radius, length=thickness
    ).named("hole")
    faces = plate.boundaries
    plate = plate.grouped(
        loaded=faces.x_max,
        symmetry_x=faces.x_min,
        symmetry_y=faces.y_min,
        hole=plate.select(faces=lambda face: face.kind == "cylinder"),
    )
    plate.save(f"kirsch_plate_hole_stress{suffix}.step")
