"""Geometry of the OneSimulation example "RLGC parameters, impedance and attenuation of a coaxial cable".

Builds a 1 mm long piece of a coaxial cable with the CAD objects of OneSimulation: an inner copper
conductor of 0.45 mm radius, a polyethylene dielectric to 1.47 mm and a copper shield to 1.62 mm.
Files: coax_rlgc_frequency_sweep.step (the whole cross-section), coax_rlgc_frequency_sweep_inner.step,
coax_rlgc_frequency_sweep_dielectric.step and coax_rlgc_frequency_sweep_shield.step. The example
computes on the cross-section z = 0.5 mm.
"""

from onesim.cad.d3 import AnnularCylinder, Cylinder

mm = 1e-3
inner, dielectric, shield, length = 0.45 * mm, 1.47 * mm, 1.62 * mm, 1.0 * mm
Cylinder(radius=shield, length=length).save("coax_rlgc_frequency_sweep.step")
Cylinder(radius=inner, length=length).save("coax_rlgc_frequency_sweep_inner.step")
AnnularCylinder(inner_radius=inner, outer_radius=dielectric, length=length).save(
    "coax_rlgc_frequency_sweep_dielectric.step"
)
AnnularCylinder(inner_radius=dielectric, outer_radius=shield, length=length).save(
    "coax_rlgc_frequency_sweep_shield.step"
)
