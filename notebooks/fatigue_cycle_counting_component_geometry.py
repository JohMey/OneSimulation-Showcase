"""Geometry of the OneSimulation example "Cycle counting and fatigue damage in a flat specimen".

Builds one quarter of the flat tensile specimen (thickness 6.25 mm) with the CAD objects of
OneSimulation and writes fatigue_cycle_counting_component.step: a 10 mm x 100 mm strip, cut back
to the 6.25 mm half-width of the central section and rounded by the fillet of radius 12.5 mm.
The face groups are symmetry_x, symmetry_y, grip, fillet and free (central-section edge, fillet
and grip edge).
"""

from math import sqrt

from onesim.cad.d3 import Box, Cylinder

half_width, half_neck, half_length, fillet_radius, thickness = (
    0.010,
    0.00625,
    0.100,
    0.0125,
    0.00625,
)
neck_end = 0.050 - sqrt(
    fillet_radius**2 - (fillet_radius - (half_width - half_neck)) ** 2
)

quarter = Box(length=half_width, width=half_length, height=thickness)
quarter = quarter - Box(
    length=half_width - half_neck,
    width=neck_end,
    height=thickness,
    origin=(half_neck, 0.0, 0.0),
).named("beside")
quarter = quarter - Cylinder(
    radius=fillet_radius,
    length=thickness,
    origin=(half_neck + fillet_radius, neck_end, 0.0),
).named("fillet")
faces = quarter.boundaries
round_face = quarter.select(faces=lambda face: face.kind == "cylinder")
straight_edge = quarter.select(
    faces=lambda face: face.kind == "plane"
    and abs(face.normal[0]) > 0.5
    and face.center[0] > 0.0
)
quarter = quarter.grouped(
    symmetry_x=faces.x_min,
    symmetry_y=faces.y_min,
    grip=faces.y_max,
    fillet=round_face,
    free=round_face | straight_edge,
)
quarter.save("fatigue_cycle_counting_component.step")
