"""Geometry of the OneSimulation example "Electric sensor: permittivity sweep over two objects in a box".

Builds the air box of the sensor, 5 m x 3 m and 1 m high, with the CAD objects of OneSimulation and
writes electric_sensor_permittivity_imaging.step with the face groups top (the driven electrode,
z = 1 m), bottom (the grounded electrode, z = 0) and walls. The two objects standing on the bottom
electrode go to one file each: a rectangular bar (1.0 m x 0.6 m, 0.8 m high, centred at x = 1.5 m,
y = 1.5 m) to electric_sensor_permittivity_imaging_bar.step and a disc (radius 0.4 m, 0.5 m high,
centred at x = 3.5 m, y = 1.5 m) to electric_sensor_permittivity_imaging_disc.step.
"""

from onesim.cad.d3 import Box, Cylinder

box = Box(length=5.0, width=3.0, height=1.0)
faces = box.boundaries
box.grouped(
    top=faces.z_max,
    bottom=faces.z_min,
    walls=faces.x_min | faces.x_max | faces.y_min | faces.y_max,
).save("electric_sensor_permittivity_imaging.step")
Box(length=1.0, width=0.6, height=0.8, origin=(1.0, 1.2, 0.0)).save(
    "electric_sensor_permittivity_imaging_bar.step"
)
Cylinder(radius=0.4, length=0.5, origin=(3.5, 1.5, 0.0)).save(
    "electric_sensor_permittivity_imaging_disc.step"
)
