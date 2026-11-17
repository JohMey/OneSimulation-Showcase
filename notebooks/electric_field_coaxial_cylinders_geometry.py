"""Geometry of the OneSimulation example "Electric field between coaxial cylinders".

Builds the dielectric between an inner conductor of radius 10 mm and an outer conductor of radius
30 mm, 20 mm long, with the CAD objects of OneSimulation and writes
electric_field_coaxial_cylinders.step. The faces keep their names inner_wall, outer_wall, bottom
and top in the file.
"""

from onesim.cad.d3 import AnnularCylinder

dielectric = AnnularCylinder(inner_radius=0.010, outer_radius=0.030, length=0.020)
dielectric.save("electric_field_coaxial_cylinders.step")
