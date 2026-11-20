"""Geometry of the OneSimulation example "Static field of a Halbach rotor".

Builds one eighth of the air around a finite four-pole Halbach rotor with the CAD objects of
OneSimulation: the part x > 0, y > 0, z > 0 of a ball of 150 mm radius, whose faces form the groups
planes (x = 0 and y = 0), midplane (z = 0, the mid-plane of the rotor) and outer, written to
halbach_rotor_field.step. The eighth of the magnet ring (inner radius 20 mm, outer radius 30 mm,
40 mm long, so 20 mm above the mid-plane) is written to halbach_rotor_field_magnet.step.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Sphere

mm = 1e-3
air, inner, outer, half = 150 * mm, 20 * mm, 30 * mm, 20 * mm
eighth = Box(length=air, width=air, height=air).named("cut")
space = Sphere(radius=air).named("ball") & eighth
faces = space.boundaries
space.grouped(
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    midplane=faces["cut.z_min"],
    outer=space.select(faces=lambda face: face.id.startswith("ball")),
).save("halbach_rotor_field.step")
(AnnularCylinder(inner_radius=inner, outer_radius=outer, length=half) & eighth).save(
    "halbach_rotor_field_magnet.step"
)
