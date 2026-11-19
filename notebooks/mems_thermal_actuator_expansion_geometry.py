"""Geometry of the OneSimulation example "Thermal expansion in a MEMS device".

Builds a meander of four hairpins etched from a 10 µm copper-beryllium layer with the CAD objects of
OneSimulation: legs 10 µm wide and 80 µm tall, half-ring arches (inner radius 10 µm, outer radius
20 µm) and short bars that join neighbouring hairpins at the bottom. The meander stands on its
bottom faces in the x-y plane and is 10 µm deep in z. Writes mems_thermal_actuator_expansion.step
with the face groups base (the bonded bottom faces at y = 0) and front (z = 10 µm).
"""

from onesim.cad.d3 import AnnularCylinder, Box, UnionFromObjects

um = 1e-6
width, height, depth, inner, outer, pitch, hairpins = (
    10 * um,
    80 * um,
    10 * um,
    10 * um,
    20 * um,
    50 * um,
    4,
)
first = -(hairpins * pitch - pitch + 4 * width) / 2

parts = []
for i in range(hairpins):
    left = first + i * pitch
    centre = left + 2 * width
    parts += [
        Box(length=width, width=height, height=depth, origin=(x, 0.0, 0.0))
        for x in (left, left + 3 * width)
    ]
    parts.append(
        AnnularCylinder(
            inner_radius=inner,
            outer_radius=outer,
            length=depth,
            origin=(centre, height, 0.0),
        )
        & Box(
            length=2 * outer,
            width=outer,
            height=depth,
            origin=(centre - outer, height, 0.0),
        )
    )
    if i < hairpins - 1:
        parts.append(
            Box(
                length=width,
                width=width,
                height=depth,
                origin=(left + 4 * width, 0.0, 0.0),
            )
        )
meander = UnionFromObjects(objects=parts)

meander = meander.grouped(
    base=meander.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-12
    ),
    front=meander.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[2] - depth) < 1e-12
    ),
)
meander.save("mems_thermal_actuator_expansion.step")
