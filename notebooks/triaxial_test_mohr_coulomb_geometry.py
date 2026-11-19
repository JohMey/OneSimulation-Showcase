"""Geometry of the OneSimulation example "Drained triaxial test on a soil sample".

Builds the standard cylindrical soil sample, 38 mm in diameter and 76 mm high, with the CAD
objects of OneSimulation. Only the half x >= 0 is kept, so that its section in the plane y = 0 is
the meridian of the axisymmetric model. The part is written to triaxial_test_mohr_coulomb.step;
the faces are named top, bottom, outer_wall (the rubber membrane) and axis.x_min (the cut through
the axis).
"""

from onesim.cad.d3 import Box, Cylinder

radius, height = 0.019, 0.076
sample = Cylinder(radius=radius, length=height) & Box(
    length=2 * radius,
    width=4 * radius,
    height=2 * height,
    origin=(0.0, -2 * radius, -height / 2),
).named("axis")
sample.save("triaxial_test_mohr_coulomb.step")
