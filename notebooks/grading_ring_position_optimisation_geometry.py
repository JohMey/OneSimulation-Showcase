"""Geometry of the OneSimulation example "Position optimisation of grading rings on an insulator".

Builds half of the air around a 1 m long rod insulator with the CAD objects of OneSimulation: a
cylinder of air (radius 3 m) from which the two metal end fittings (radius 6 cm) and two grading
rings (torus, ring radius 15 cm, tube radius 2 cm) are cut out. The faces of the high-voltage
fitting and its ring form the group hv, those of the grounded fitting and its ring the group ground,
the outer faces of the air the group outer. One file is written for each distance of the rings from
the ends of the rod, 0 to 30 cm: grading_ring_position_optimisation.step (10 cm) and
grading_ring_position_optimisation_00cm.step ... _30cm.step. The rod (radius 4 cm, dielectric) is
written to grading_ring_position_optimisation_rod.step; its end faces carry the names of the fittings
(hv.bottom, ground.top).
"""

from onesim.cad.d3 import Box, Cylinder, Torus

rod_radius, length, fitting_radius, fitting_length = 0.04, 1.0, 0.06, 0.15
ring_radius, tube_radius, air_radius = 0.15, 0.02, 3.0
half = Box(
    length=air_radius, width=2 * air_radius, height=4.0, origin=(0.0, -air_radius, -1.5)
)
air = Cylinder(radius=air_radius, length=4.0, origin=(0.0, 0.0, -1.5)).named("air")
for offset in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
    hv = Cylinder(
        radius=fitting_radius, length=fitting_length, origin=(0.0, 0.0, length)
    ) | Torus(
        major_radius=ring_radius,
        minor_radius=tube_radius,
        origin=(0.0, 0.0, length - offset),
    )
    ground = Cylinder(
        radius=fitting_radius, length=fitting_length, origin=(0.0, 0.0, -fitting_length)
    ) | Torus(
        major_radius=ring_radius, minor_radius=tube_radius, origin=(0.0, 0.0, offset)
    )
    region = (air - hv.named("hv") - ground.named("ground")) & half
    region = region.grouped(
        hv=region.select(faces=lambda face: face.id.startswith("hv")),
        ground=region.select(faces=lambda face: face.id.startswith("ground")),
        outer=region.select(faces=lambda face: face.id.startswith("air")),
    )
    region.save(
        "grading_ring_position_optimisation.step"
        if offset == 0.10
        else f"grading_ring_position_optimisation_{round(100 * offset):02d}cm.step"
    )
rod = (
    Cylinder(
        radius=rod_radius,
        length=length + 2 * fitting_length,
        origin=(0.0, 0.0, -fitting_length),
    )
    - Cylinder(
        radius=fitting_radius, length=fitting_length, origin=(0.0, 0.0, length)
    ).named("hv")
    - Cylinder(
        radius=fitting_radius, length=fitting_length, origin=(0.0, 0.0, -fitting_length)
    ).named("ground")
)
(rod & half).save("grading_ring_position_optimisation_rod.step")
