"""Geometry of the OneSimulation example "Cascaded rectangular cavity filter".

Builds with the CAD objects of OneSimulation the air of a WR-90 waveguide (22.86 mm x 10.16 mm,
axis along x, broad side along y, narrow side along z) that holds three rectangular cavities in a
row, separated from each other and from the feeding guides by metal walls 2 mm thick. Each wall
has a centred coupling slot over the full height of the guide; its edges are rounded with a
radius of 0.5 mm, as a milling cutter leaves them. The slot widths and the cavity
lengths realize a three-pole Chebyshev bandpass filter (0.1 dB ripple, 9.8 to 10.2 GHz): coupling
coefficients and external quality factor from the low-pass prototype, turned into slots and
lengths with the impedance-inverter formulas of Matthaei, Young and Jones, and refined on the
H-plane section for the 2 mm walls and the rounded edges. 10 mm of guide lead to the first and from the last wall.

Files: cascaded_cavity_filter.step (the filter, faces port_in, port_out, metal and, within metal,
the coupling walls as walls) and
cascaded_cavity_filter_cavity.step (the middle cavity closed, all faces in metal).
"""

import numpy as np
from onesim.cad.d3 import Box

a, b, wall, lead, radius = 0.02286, 0.01016, 0.002, 0.010, 0.0005
slots = (0.01131, 0.00780, 0.00780, 0.01131)
spacings = (0.01804, 0.01960, 0.01804)
centres = lead + wall / 2 + np.concatenate(([0.0], np.cumsum(spacings)))
air = Box(length=centres[-1] + wall / 2 + lead, width=a, height=b).named("guide")
for x, width in zip(centres, slots):
    for y, side in ((-0.001, "y_max"), ((a + width) / 2, "y_min")):
        block = Box(
            length=wall,
            width=(a - width) / 2 + 0.001,
            height=b + 0.002,
            origin=(x - wall / 2, y, -0.001),
        )
        block = block.fillet(
            radius,
            edges=block.select(
                edges=lambda edge, side=side: set(edge.faces)
                in ({"x_min", side}, {"x_max", side})
            ),
        )
        air = air - block.named("wall")
faces = air.boundaries
walls = air.select(faces=lambda face: face.id.startswith("wall."))
air.grouped(
    port_in=faces["guide.x_min"],
    port_out=faces["guide.x_max"],
    walls=walls,
    metal=walls
    | faces["guide.y_min"]
    | faces["guide.y_max"]
    | faces["guide.z_min"]
    | faces["guide.z_max"],
).save("cascaded_cavity_filter.step")
closed = Box(length=spacings[1] - wall, width=a, height=b)
closed.grouped(metal=closed.boundaries.all).save("cascaded_cavity_filter_cavity.step")
