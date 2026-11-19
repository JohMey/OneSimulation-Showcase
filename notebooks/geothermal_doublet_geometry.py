"""Geometry of the OneSimulation example "Thermal breakthrough of a geothermal doublet".

Builds one half of a confined aquifer, 3000 m long, 1000 m wide (half of 2000 m) and 50 m thick,
cut by the vertical symmetry plane through the two wells, with the CAD objects of OneSimulation.
The injection and the production well, radius 0.5 m (screen with gravel pack) and 600 m apart, penetrate the whole
thickness and are cut out as half cylinders. The part is written to geothermal_doublet.step; its
faces are named injector.outer_wall, producer.outer_wall, y_min (symmetry plane) and the box faces
x_min, x_max, y_max, z_min, z_max.
"""

from onesim.cad.d3 import Box, Cylinder

length, half_width, thickness, spacing, radius = 3000.0, 1000.0, 50.0, 600.0, 0.5
well = lambda x, name: Cylinder(
    radius=radius, length=thickness + 2.0, origin=(x, 0.0, -1.0)
).named(name)
aquifer = Box(
    length=length, width=half_width, height=thickness, origin=(-length / 2, 0.0, 0.0)
)
(aquifer - well(-spacing / 2, "injector") - well(spacing / 2, "producer")).save(
    "geothermal_doublet.step"
)
