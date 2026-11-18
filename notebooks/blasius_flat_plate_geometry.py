"""Geometry of the OneSimulation example "Laminar flat-plate boundary layer (Blasius)".

Builds the slab of air above the plate, 1 m along the plate, 0.25 m high and 0.05 m across the
span, with the CAD objects of OneSimulation and writes blasius_flat_plate.step. The plate is the
face y_min, the inflow x_min; the faces keep their names x_min, x_max, y_min, y_max, z_min and
z_max in the file. The example computes on the cross-section z = 0.025 m.
"""

from onesim.cad.d3 import Box

slab = Box(length=1.0, width=0.25, height=0.05)
slab.save("blasius_flat_plate.step")
