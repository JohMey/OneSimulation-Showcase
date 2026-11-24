"""Geometry of the OneSimulation example "Lumping a side-branch resonator into an impedance".

Builds, with the CAD objects of OneSimulation, half of the air (cut in the symmetry plane y = 0) of
three models: a duct of radius 25 mm and length 500 mm with a Helmholtz resonator on its side - a
neck of radius 10 mm and length 30 mm and a cavity of radius 30 mm and length 50 mm -
(resonator_impedance_lumping.step); the resonator alone from 5 mm above the duct wall
(resonator_impedance_lumping_resonator.step); and the duct with only a 5 mm stub of the neck
(resonator_impedance_lumping_duct.step). The faces are named inlet, outlet, port (the neck section
5 mm above the duct wall), symmetry and walls where they exist.
"""

from onesim.cad.d3 import Box, Cylinder

duct_radius, duct_length, neck_radius, neck_length, cavity_radius, cavity_length = (
    0.025,
    0.5,
    0.010,
    0.030,
    0.030,
    0.050,
)
middle, stub = duct_length / 2, 0.005
across = dict(axis=(1.0, 0.0, 0.0))
duct = Cylinder(radius=duct_radius, length=duct_length)
neck = Cylinder(
    radius=neck_radius,
    length=neck_length + 0.01,
    origin=(duct_radius - 0.01, 0.0, middle),
    **across,
)
cavity = Cylinder(
    radius=cavity_radius,
    length=cavity_length,
    origin=(duct_radius + neck_length, 0.0, middle),
    **across,
)
lower_half = Box(length=0.4, width=0.2, height=0.8, origin=(-0.2, -0.2, -0.2))
plane = lambda normal, coordinate, value: (
    lambda face: face.kind == "plane"
    and abs(abs(face.normal[normal]) - 1) < 1e-9
    and abs(face.center[coordinate] - value) < 1e-9
)
models = {
    "resonator_impedance_lumping": duct | neck | cavity,
    "resonator_impedance_lumping_resonator": (
        Cylinder(
            radius=neck_radius,
            length=neck_length - stub,
            origin=(duct_radius + stub, 0.0, middle),
            **across,
        )
        | cavity
    ),
    "resonator_impedance_lumping_duct": duct
    | Cylinder(
        radius=neck_radius,
        length=stub + 0.01,
        origin=(duct_radius - 0.01, 0.0, middle),
        **across,
    ),
}
for name, solid in models.items():
    half = solid - lower_half
    groups = dict(
        inlet=plane(2, 2, 0.0),
        outlet=plane(2, 2, duct_length),
        port=plane(0, 0, duct_radius + stub),
        symmetry=plane(1, 1, 0.0),
    )
    found = {}
    for group, test in groups.items():
        try:
            found[group] = half.select(faces=test)
        except ValueError:
            pass
    named = lambda face: any(
        test(face) for group, test in groups.items() if group in found
    )
    found["walls"] = half.select(faces=lambda face: not named(face))
    half.grouped(**found).save(f"{name}.step")
