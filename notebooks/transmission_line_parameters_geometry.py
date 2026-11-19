"""Geometry of the OneSimulation example "Transmission line parameters: coaxial cable and twin lead".

Builds 1 mm long pieces of two transmission lines with the CAD objects of OneSimulation.
Coaxial cable (RG-58 class): inner conductor of radius 0.45 mm, polyethylene dielectric to 1.47 mm,
copper shield 0.15 mm thick. Files: transmission_line_parameters.step (the whole cable, the domain
of the magnetic model), _coax_dielectric.step (the electrostatic domain), _coax_inner.step and
_coax_shield.step (the two conductors).
Twin lead: two wires of radius 0.5 mm, 7.5 mm apart, in air inside a cylinder of 375 mm radius.
Files: _twin.step (the air cylinder with the wires, the magnetic domain), _twin_air.step (the air
with the wires cut out, the electrostatic domain), _twin_plus.step and _twin_minus.step (the wires).
"""

from onesim.cad.d3 import AnnularCylinder, Cylinder

mm = 1e-3
inner, dielectric, shield, length = 0.45 * mm, 1.47 * mm, 1.62 * mm, 1.0 * mm
Cylinder(radius=shield, length=length).save("transmission_line_parameters.step")
AnnularCylinder(inner_radius=inner, outer_radius=dielectric, length=length).save(
    "transmission_line_parameters_coax_dielectric.step"
)
Cylinder(radius=inner, length=length).save(
    "transmission_line_parameters_coax_inner.step"
)
AnnularCylinder(inner_radius=dielectric, outer_radius=shield, length=length).save(
    "transmission_line_parameters_coax_shield.step"
)

wire, spacing, outer = 0.5 * mm, 7.5 * mm, 375 * mm
plus = Cylinder(radius=wire, length=length, origin=(spacing / 2, 0.0, 0.0))
minus = Cylinder(radius=wire, length=length, origin=(-spacing / 2, 0.0, 0.0))
Cylinder(radius=outer, length=length).save("transmission_line_parameters_twin.step")
(
    Cylinder(radius=outer, length=length) - plus.named("plus") - minus.named("minus")
).save("transmission_line_parameters_twin_air.step")
plus.save("transmission_line_parameters_twin_plus.step")
minus.save("transmission_line_parameters_twin_minus.step")
