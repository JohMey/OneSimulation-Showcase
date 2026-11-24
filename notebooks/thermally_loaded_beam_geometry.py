"""Geometry of the OneSimulation example "Thermally loaded beam".

Builds a steel beam, 1.5 m long with a 60 mm x 100 mm rectangular cross-section centred on the
x axis, with the CAD objects of OneSimulation and writes thermally_loaded_beam.step. The faces
keep their names x_min, x_max (the two ends), y_min, y_max (left and right) and z_min, z_max
(bottom and top).
"""

from onesim.cad.d3 import Box

length, breadth, depth = 1.5, 0.060, 0.100
beam = Box(
    length=length, width=breadth, height=depth, origin=(0.0, -breadth / 2, -depth / 2)
)
beam.save("thermally_loaded_beam.step")
