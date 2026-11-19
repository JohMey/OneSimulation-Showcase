"""Geometry of the OneSimulation example "Optimizing the currents of a coil system for a uniform field".

Builds, with the CAD objects of OneSimulation, the upper half (z > 0) of the space around a system
of six coaxial coils placed symmetrically about the plane z = 0: a half cylinder of 1 m radius and
1 m height on the side x > 0, whose meridian section is the axisymmetric domain, written to
coil_current_optimization_axisymmetric.step. The three coils above the midplane (inner radius
0.10 m, outer radius 0.12 m, 20 mm high, centred at z = 0.05, 0.15 and 0.25 m) go to
coil_current_optimization_axisymmetric_coil_1.step to _coil_3.step, each cut to the side x > 0.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

extent, inner, outer, height = 1.0, 0.10, 0.12, 0.02
side = Box(length=extent, width=2 * extent, height=extent, origin=(0.0, -extent, 0.0))
(Cylinder(radius=extent, length=extent) & side).save(
    "coil_current_optimization_axisymmetric.step"
)
for index, centre in enumerate((0.05, 0.15, 0.25), start=1):
    coil = (
        AnnularCylinder(
            inner_radius=inner,
            outer_radius=outer,
            length=height,
            origin=(0.0, 0.0, centre - height / 2),
        )
        & side
    )
    coil.save(f"coil_current_optimization_axisymmetric_coil_{index}.step")
