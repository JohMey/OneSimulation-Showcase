"""Geometry of the OneSimulation example "Stress concentration at a superelliptic hole".

Builds one quarter of a square plate (half-width 200 mm, 10 mm thick) with a central superelliptic
hole |x/a|^n + |y/a|^n = 1, a = 10 mm, with the CAD objects of OneSimulation, for the exponents
n = 2, 3, 4, 6 and 10. The plate with n = 4 is written to superellipse_hole_stress_concentration.step,
the others to superellipse_hole_stress_concentration_n2.step, _n3, _n6 and _n10. The hole outline is
a polygon through 480 points of the curve. The faces are named hole, loaded (x = 200 mm),
symmetry_x and symmetry_y.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Prism

radius, half_width, thickness = 0.010, 0.200, 0.010
angle = np.linspace(0.0, 2 * np.pi, 480, endpoint=False)
for exponent, suffix in ((4, ""), (2, "_n2"), (3, "_n3"), (6, "_n6"), (10, "_n10")):
    outline = radius * np.column_stack(
        (
            np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** (2 / exponent),
            np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** (2 / exponent),
        )
    )
    hole = Prism(
        profile=Profile.polygon([tuple(p) for p in outline], name="hole"),
        height=thickness,
    )
    plate = Box(length=half_width, width=half_width, height=thickness) - hole
    faces = plate.boundaries
    plate = plate.grouped(
        loaded=faces.x_max,
        symmetry_x=faces.x_min,
        symmetry_y=faces.y_min,
        hole=plate.select(
            faces=lambda face: abs(face.normal[2]) < 0.5
            and np.hypot(*face.center[:2]) < 1.5 * radius
        ),
    )
    plate.save(f"superellipse_hole_stress_concentration{suffix}.step")
