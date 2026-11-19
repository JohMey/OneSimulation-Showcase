"""Geometry of the OneSimulation example "Submodel with plasticity at a hole in a plate".

Builds one eighth of a steel plate with a central hole (plate 100 mm wide, 200 mm long and 10 mm
thick, hole diameter 20 mm), cut by its three symmetry planes, with the CAD objects of
OneSimulation, and the submodel: the block of 25 mm x 25 mm around the hole cut from the same
eighth. The plate is written to plate_hole_plastic_submodel.step, the submodel to
plate_hole_plastic_submodel_detail.step. In both the faces keep the box names x_min, y_min,
z_min (symmetry planes), x_max, y_max, z_max, and the hole wall is hole.outer_wall; in the
submodel x_max and y_max are the cut faces that take the displacements of the plate model.
"""

from onesim.cad.d3 import Box, Cylinder

half_width, half_length, half_thickness, radius, detail = (
    0.050,
    0.100,
    0.005,
    0.010,
    0.025,
)
hole = lambda: Cylinder(
    radius=radius, length=4 * half_thickness, origin=(0.0, 0.0, -half_thickness)
).named("hole")
(Box(length=half_width, width=half_length, height=half_thickness) - hole()).save(
    "plate_hole_plastic_submodel.step"
)
(Box(length=detail, width=detail, height=half_thickness) - hole()).save(
    "plate_hole_plastic_submodel_detail.step"
)
