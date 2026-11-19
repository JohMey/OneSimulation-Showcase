"""Geometry of the OneSimulation example "Eddy currents in a cylinder inside a long coil".

Builds, with the CAD objects of OneSimulation, a 10 mm slice of a long coil (inner radius 15 mm,
outer radius 17 mm) around a conducting cylinder of 10 mm radius, inside an air region of 20 mm
radius; each part is cut to the side x > 0, so that its section with the plane y = 0 is the
meridian half-plane. Files: eddy_currents_cylinder_in_coil.step (the air region, the domain),
eddy_currents_cylinder_in_coil_cylinder.step and eddy_currents_cylinder_in_coil_winding.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

height, radius, inner, outer, edge = 0.010, 0.010, 0.015, 0.017, 0.020
side = Box(length=edge, width=2 * edge, height=height, origin=(0.0, -edge, 0.0))
(Cylinder(radius=edge, length=height) & side).save(
    "eddy_currents_cylinder_in_coil.step"
)
(Cylinder(radius=radius, length=height) & side).save(
    "eddy_currents_cylinder_in_coil_cylinder.step"
)
(AnnularCylinder(inner_radius=inner, outer_radius=outer, length=height) & side).save(
    "eddy_currents_cylinder_in_coil_winding.step"
)
