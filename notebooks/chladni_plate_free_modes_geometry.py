"""Geometry of the OneSimulation example "Chladni figures of a free square plate".

Builds a square brass plate, 240 mm wide and 2.4 mm thick, with the CAD objects of OneSimulation and
writes chladni_plate_free_modes.step. The faces keep their names x_min, x_max, y_min, y_max, z_min
and z_max in the file.
"""

from onesim.cad.d3 import Box

plate = Box(length=0.240, width=0.240, height=0.0024, origin=(-0.120, -0.120, 0.0))
plate.save("chladni_plate_free_modes.step")
