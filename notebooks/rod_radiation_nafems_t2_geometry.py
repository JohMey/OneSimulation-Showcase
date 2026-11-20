"""Geometry of the OneSimulation example "Steady conduction in a rod with a radiating end (NAFEMS T2)".

Builds the rod, 0.1 m long along x with a 10 mm x 10 mm cross-section, with the CAD objects of
OneSimulation and writes rod_radiation_nafems_t2.step. The held end is named hot_end, the
radiating end radiating_end and the four lateral faces form the group sides.
"""

from onesim.cad.d3 import Box

rod = Box(length=0.1, width=0.01, height=0.01)
faces = rod.boundaries
rod = rod.grouped(
    hot_end=faces.x_min,
    radiating_end=faces.x_max,
    sides=faces.y_min | faces.y_max | faces.z_min | faces.z_max,
)
rod.save("rod_radiation_nafems_t2.step")
