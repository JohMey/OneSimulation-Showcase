"""Geometry of the OneSimulation example "Electric shielding by a thin conductive shell".

Builds a conducting plate, 4 m x 4 m and 0.1 m deep, with the CAD objects of OneSimulation and writes
conductive_shell_current_shielding.step; its top face is the group terminal, its bottom face the group
ground and its two sides the group insulated. A tube of 0.25 m outer radius and 1 cm wall sits at the
centre: its wall goes to conductive_shell_current_shielding_wall.step, its interior to
conductive_shell_current_shielding_core.step and the cylinder up to the middle of the wall to
conductive_shell_current_shielding_midline.step. The example computes on the cross-section z = 0.05 m.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

side, depth, radius, wall = 4.0, 0.1, 0.25, 0.01
plate = Box(length=side, width=side, height=depth, origin=(-side / 2, -side / 2, 0.0))
faces = plate.boundaries
plate = plate.grouped(
    terminal=faces.y_max, ground=faces.y_min, insulated=faces.x_min | faces.x_max
)
plate.save("conductive_shell_current_shielding.step")
AnnularCylinder(inner_radius=radius - wall, outer_radius=radius, length=depth).save(
    "conductive_shell_current_shielding_wall.step"
)
Cylinder(radius=radius - wall, length=depth).save(
    "conductive_shell_current_shielding_core.step"
)
Cylinder(radius=radius - wall / 2, length=depth).save(
    "conductive_shell_current_shielding_midline.step"
)
