"""Geometry of the OneSimulation example "Inductive heating of a copper cylinder".

Builds, with the CAD objects of OneSimulation, a 10 mm slice of a long coil (inner radius 15 mm,
outer radius 17 mm) around a copper cylinder of 10 mm radius, inside an air region of 20 mm
radius; each part is cut to the side x > 0, so that its section with the plane y = 0 is the
meridian half-plane. Files: inductive_heating_copper_cylinder.step (the air region, the domain),
inductive_heating_copper_cylinder_cylinder.step and inductive_heating_copper_cylinder_winding.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

height, radius, inner, outer, edge = 0.010, 0.010, 0.015, 0.017, 0.020
side = Box(length=edge, width=2 * edge, height=height, origin=(0.0, -edge, 0.0))
(Cylinder(radius=edge, length=height) & side).save(
    "inductive_heating_copper_cylinder.step"
)
(Cylinder(radius=radius, length=height) & side).save(
    "inductive_heating_copper_cylinder_cylinder.step"
)
(AnnularCylinder(inner_radius=inner, outer_radius=outer, length=height) & side).save(
    "inductive_heating_copper_cylinder_winding.step"
)
