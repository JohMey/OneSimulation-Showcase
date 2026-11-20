"""Geometry of the OneSimulation example "Composite thermal barrier: resolved layers and lumped resistors".

Builds with the CAD objects of OneSimulation a copper heater block (30 mm x 30 mm, 10 mm thick) on a
composite barrier of four layers, each 1 mm thick and of the heater's footprint, on an aluminium
base plate (60 mm x 60 mm, 5 mm thick) that is cooled from below.

Files: composite_thermal_barrier_lumped.step holds the whole stack as one solid with the face
group bottom (the cooled face); composite_thermal_barrier_lumped_<name>.step
hold the regions of the heater and of the four layers (mica, ceramic_paper, glass_fibre,
graphite_foil, from the plate upwards). For the lumped model, composite_thermal_barrier_lumped_plate.step
and composite_thermal_barrier_lumped_heater_on_plate.step hold the plate and the heater placed
directly on it, the barrier being replaced by a thermal resistance between them.
"""

from onesim.cad.d3 import Box

plate_side, plate, side, layer, heater = 0.06, 0.005, 0.03, 0.001, 0.010
corner = (plate_side - side) / 2
base = Box(length=plate_side, width=plate_side, height=plate)
stack = base | Box(
    length=side, width=side, height=4 * layer + heater, origin=(corner, corner, plate)
)
faces = stack.boundaries
stack.grouped(bottom=faces.z_min).save("composite_thermal_barrier_lumped.step")
for index, name in enumerate(("mica", "ceramic_paper", "glass_fibre", "graphite_foil")):
    Box(
        length=side,
        width=side,
        height=layer,
        origin=(corner, corner, plate + index * layer),
    ).save(f"composite_thermal_barrier_lumped_{name}.step")
Box(
    length=side, width=side, height=heater, origin=(corner, corner, plate + 4 * layer)
).save("composite_thermal_barrier_lumped_heater.step")
base.grouped(bottom=base.boundaries.z_min).save(
    "composite_thermal_barrier_lumped_plate.step"
)
Box(length=side, width=side, height=heater, origin=(corner, corner, plate)).save(
    "composite_thermal_barrier_lumped_heater_on_plate.step"
)
