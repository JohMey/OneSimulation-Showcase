"""Geometry of the OneSimulation example "Electric sensor: dielectric objects in a box".

Builds the air box of the sensor, 5 m x 3 m and 1 m high, with the CAD objects of OneSimulation and
writes dielectric_inclusion_sensor.step with the face groups top (the driven electrode, z = 1 m),
bottom (the grounded electrode, z = 0) and walls. The two dielectric objects standing on the bottom
electrode go to one file each: a rectangular bar (1.0 m x 0.6 m, 0.8 m high, centred at x = 1.5 m,
y = 1.5 m) to dielectric_inclusion_sensor_bar.step and a disc (radius 0.4 m, 0.5 m high, centred at
x = 3.5 m, y = 1.5 m) to dielectric_inclusion_sensor_disc.step. The same two objects extended to the
full height of the box, used for the exact check, go to ..._bar_column.step and ..._disc_column.step.
"""

from onesim.cad.d3 import Box, Cylinder

length, width, height = 5.0, 3.0, 1.0
box = Box(length=length, width=width, height=height)
faces = box.boundaries
box.grouped(
    top=faces.z_max,
    bottom=faces.z_min,
    walls=faces.x_min | faces.x_max | faces.y_min | faces.y_max,
).save("dielectric_inclusion_sensor.step")
for suffix, bar_height, disc_height in (("", 0.8, 0.5), ("_column", height, height)):
    Box(length=1.0, width=0.6, height=bar_height, origin=(1.0, 1.2, 0.0)).save(
        f"dielectric_inclusion_sensor_bar{suffix}.step"
    )
    Cylinder(radius=0.4, length=disc_height, origin=(3.5, 1.5, 0.0)).save(
        f"dielectric_inclusion_sensor_disc{suffix}.step"
    )
