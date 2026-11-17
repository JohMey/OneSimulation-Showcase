"""Geometry of the OneSimulation example "Periodic surface temperature wave in a semi-infinite solid".

Builds the soil column with the CAD objects of OneSimulation and writes
periodic_surface_temperature_wave.step. The column is seven penetration depths of the daily cycle
long (moist soil, diffusivity 5e-7 m^2/s: d = 11.7 cm) with a 20 mm x 20 mm cross-section. The
surface and the deep end keep their names x_min and x_max in the file; the four lateral faces form
the group sides.
"""

from math import pi, sqrt

from onesim.cad.d3 import Box

diffusivity, period = 1.0 / (2000.0 * 1000.0), 86400.0
depth = sqrt(2 * diffusivity * period / (2 * pi))
column = Box(length=7 * depth, width=0.020, height=0.020)
faces = column.boundaries
column = column.grouped(sides=faces.y_min | faces.y_max | faces.z_min | faces.z_max)
column.save("periodic_surface_temperature_wave.step")
