"""Geometry of the OneSimulation example "One-sided Halbach magnet attracting a steel plate".

Builds one quarter of a one-sided magnet and a steel plate with the CAD objects of OneSimulation.
The magnet is a row of five blocks, 10 mm x 30 mm x 10 mm each, whose magnetization turns by a
quarter turn from block to block (a Halbach array); the steel plate, 70 mm x 40 mm x 6 mm, lies
parallel to its top face. The quarter x > 0, y > 0 is cut by the two symmetry planes.

Files: halbach_magnet_steel_plate_force.step holds the air box (the modelled region) with the face
groups symmetry_x, symmetry_y and outer; halbach_magnet_steel_plate_force_block_0.step to _block_2.step
hold the quarter of the middle block and the two blocks beside it; _plate_1mm.step, _plate_2mm.step
and _plate_4mm.step hold the plate at an air gap of 1, 2 and 4 mm.
"""

from onesim.cad.d3 import Box

block, width, plate_length, plate_width, plate_thickness, air = (
    0.010,
    0.030,
    0.070,
    0.040,
    0.006,
    0.150,
)

region = Box(length=air, width=air, height=2 * air, origin=(0.0, 0.0, -air))
faces = region.boundaries
region = region.grouped(
    symmetry_x=faces.x_min,
    symmetry_y=faces.y_min,
    outer=faces.x_max | faces.y_max | faces.z_min | faces.z_max,
)
region.save("halbach_magnet_steel_plate_force.step")

for index, (start, length) in enumerate(
    ((0.0, block / 2), (block / 2, block), (3 * block / 2, block))
):
    Box(length=length, width=width / 2, height=block, origin=(start, 0.0, -block)).save(
        f"halbach_magnet_steel_plate_force_block_{index}.step"
    )

for gap in (1, 2, 4):
    Box(
        length=plate_length / 2,
        width=plate_width / 2,
        height=plate_thickness,
        origin=(0.0, 0.0, gap * 1e-3),
    ).save(f"halbach_magnet_steel_plate_force_plate_{gap}mm.step")
