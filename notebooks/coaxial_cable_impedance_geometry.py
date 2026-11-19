"""Geometry of the OneSimulation example "Characteristic impedance of a coaxial cable".

Builds a 1 mm long piece of the polyethylene dielectric of a coaxial cable with the CAD objects of
OneSimulation: inner conductor diameter 0.90 mm, shield inner diameter 2.95 mm. The concentric
cable is written to coaxial_cable_impedance.step with the faces inner_wall (inner conductor) and
outer_wall (shield); a cable whose inner conductor is displaced by 0.40 mm is written to
coaxial_cable_impedance_eccentric.step with the faces conductor.outer_wall and outer_wall.
"""

from onesim.cad.d3 import AnnularCylinder, Cylinder

inner, outer, length, offset = 0.45e-3, 1.475e-3, 1.0e-3, 0.40e-3
AnnularCylinder(inner_radius=inner, outer_radius=outer, length=length).save(
    "coaxial_cable_impedance.step"
)
eccentric = Cylinder(radius=outer, length=length) - Cylinder(
    radius=inner, length=length, origin=(offset, 0.0, 0.0)
).named("conductor")
eccentric.save("coaxial_cable_impedance_eccentric.step")
