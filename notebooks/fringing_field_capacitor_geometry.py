"""Geometry of the OneSimulation example "Fringing fields and the capacitance of a disc capacitor".

Builds one eighth of the air around a capacitor of two metal discs (radius 10 cm, thickness 5 mm,
gap 3 cm) with the CAD objects of OneSimulation: the part of an air sphere with x, y, z >= 0, with
the upper disc removed. The mid plane z = 0 replaces the lower disc. One file is written for every
radius of the air sphere: fringing_field_capacitor.step (15 cm) and
fringing_field_capacitor_21cm.step to fringing_field_capacitor_39cm.step. Every file names the
face groups terminal (the disc), midplane, symmetry and outer (the sphere).
"""

from onesim.cad.d3 import Box, Cylinder, Sphere

disc_radius, thickness, gap = 0.10, 0.005, 0.03
for radius in (0.15, 0.21, 0.27, 0.33, 0.39):
    air = Sphere(radius=radius) & Box(length=radius, width=radius, height=radius).named(
        "cut"
    )
    air = air - Cylinder(
        radius=disc_radius, length=thickness, origin=(0.0, 0.0, gap / 2)
    ).named("disc")
    faces = air.boundaries
    air = air.grouped(
        terminal=faces.disc,
        midplane=faces["cut.z_min"],
        symmetry=faces["cut.x_min"] | faces["cut.y_min"],
        outer=faces.surface,
    )
    air.save(
        "fringing_field_capacitor.step"
        if radius == 0.15
        else f"fringing_field_capacitor_{100 * radius:.0f}cm.step"
    )
