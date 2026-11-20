"""Geometry of the OneSimulation example "Thin low-permittivity gap in a dielectric capacitor".

Builds the dielectric between the plates of a capacitor of two parallel plates, 80 mm x 80 mm and 5 mm deep,
with the CAD objects of OneSimulation and writes thin_low_permittivity_gap_capacitor.step; its top face is
the group terminal (the upper plate), its bottom face the group ground (the lower plate) and its two
sides the group insulated. A circular gap of 5 mm outer radius sits at the centre, 0.1, 0.25, 0.5 or
1 mm wide. For each gap width, in micrometres, the gap goes to
thin_low_permittivity_gap_capacitor_wall_<um>.step, the interior to thin_low_permittivity_gap_capacitor_core_<um>.step
and the cylinder up to the middle of the gap to thin_low_permittivity_gap_capacitor_midline_<um>.step.
The example computes on the cross-section z = 2.5 mm.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

side, depth, radius = 0.08, 0.005, 0.005
gap = Box(length=side, width=side, height=depth, origin=(-side / 2, -side / 2, 0.0))
faces = gap.boundaries
gap = gap.grouped(
    terminal=faces.y_max, ground=faces.y_min, insulated=faces.x_min | faces.x_max
)
gap.save("thin_low_permittivity_gap_capacitor.step")
for micrometres in (100, 250, 500, 1000):
    wall = micrometres * 1e-6
    AnnularCylinder(inner_radius=radius - wall, outer_radius=radius, length=depth).save(
        f"thin_low_permittivity_gap_capacitor_wall_{micrometres}.step"
    )
    Cylinder(radius=radius - wall, length=depth).save(
        f"thin_low_permittivity_gap_capacitor_core_{micrometres}.step"
    )
    Cylinder(radius=radius - wall / 2, length=depth).save(
        f"thin_low_permittivity_gap_capacitor_midline_{micrometres}.step"
    )
