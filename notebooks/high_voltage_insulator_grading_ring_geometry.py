"""Geometry of the OneSimulation example "Surface field of a high-voltage insulator with a grading ring".

Builds a 30-degree sector of a 110 kV composite long-rod insulator and the air around it with the
CAD objects of OneSimulation. The insulator hangs from a grounded crossarm: a 1 m silicone housing
(core radius 12 mm) with 16 sheds of alternating radius (60 mm and 45 mm), metal end fittings at
both ends, and on the high-voltage end a grading ring (torus, major radius 120 mm, tube radius
16 mm). The modelled sector of the air is a cylinder of 1.5 m radius whose top face is the crossarm.

Files: high_voltage_insulator_grading_ring.step holds the air with fittings and ring cut out, with
the face groups crossarm, live (high-voltage fitting and ring), grounded_fitting, symmetry and
outer; high_voltage_insulator_grading_ring_no_ring.step the same without the ring;
high_voltage_insulator_grading_ring_housing.step holds the sector of the silicone housing.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder, Torus

core, length, fitting_radius, fitting_length = 0.012, 1.0, 0.025, 0.10
air_radius, air_bottom, air_top = 1.5, -1.6, 0.5 + fitting_length
sector = np.radians(30.0)

half_plane = Box(
    length=4 * air_radius,
    width=2 * air_radius,
    height=4.0,
    origin=(-2 * air_radius, 0.0, -2.0),
)
in_sector = (
    lambda solid: solid
    & half_plane.named("cut_0")
    & half_plane.rotated((0.0, 0.0, 1.0), sector - np.pi).named("cut_30")
)

sheds = [
    Cylinder(
        radius=0.060 if k % 2 == 0 else 0.045,
        length=0.004,
        origin=(0.0, 0.0, -0.45 + k * 0.06),
    )
    for k in range(16)
]
housing = Cylinder(radius=core, length=length, origin=(0.0, 0.0, -0.5))
for shed in sheds:
    housing = housing | shed
live_fitting = Cylinder(
    radius=fitting_radius,
    length=fitting_length,
    origin=(0.0, 0.0, -0.5 - fitting_length),
)
grounded_fitting = Cylinder(
    radius=fitting_radius, length=fitting_length, origin=(0.0, 0.0, 0.5)
)
ring = Torus(major_radius=0.12, minor_radius=0.016, origin=(0.0, 0.0, -0.44))

for name, live in (
    ("high_voltage_insulator_grading_ring.step", live_fitting | ring),
    ("high_voltage_insulator_grading_ring_no_ring.step", live_fitting),
):
    air = in_sector(
        Cylinder(
            radius=air_radius,
            length=air_top - air_bottom,
            origin=(0.0, 0.0, air_bottom),
        )
        - live.named("live")
        - grounded_fitting.named("grounded")
    )
    faces = air.boundaries
    air = air.grouped(
        crossarm=faces.top,
        live=air.select(faces=lambda face: face.id.startswith("live.")),
        grounded_fitting=air.select(faces=lambda face: face.id.startswith("grounded.")),
        symmetry=faces["cut_0.y_min"]
        | air.select(faces=lambda face: face.id.startswith("cut_30.")),
        outer=faces.bottom | faces.outer_wall,
    )
    air.save(name)
in_sector(housing).save("high_voltage_insulator_grading_ring_housing.step")
