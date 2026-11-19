"""Geometry of the OneSimulation example "Flow past a cylinder: the DFG benchmark 2D-1".

Builds the channel, 2.2 m long and 0.41 m high, with the cylinder of diameter 0.1 m centred at
(0.2, 0.2) m cut out of it, as a slab 0.1 m thick with the CAD objects of OneSimulation and writes
flow_past_cylinder_dfg_benchmark.step. The faces are grouped as inlet (x = 0), outlet (x = 2.2 m),
walls (y = 0 and y = 0.41 m) and cylinder. The example computes on the cross-section z = 0.05 m,
whose sides keep the names x_min, x_max, y_min, y_max and cylinder.outer_wall.
"""

from onesim.cad.d3 import Box, Cylinder

thickness = 0.1
channel = Box(length=2.2, width=0.41, height=thickness) - Cylinder(
    radius=0.05, length=thickness, origin=(0.2, 0.2, 0.0)
).named("cylinder")
faces = channel.boundaries
channel = channel.grouped(
    inlet=faces.x_min,
    outlet=faces.x_max,
    walls=faces.y_min | faces.y_max,
    cylinder=channel.select(faces=lambda face: face.kind == "cylinder"),
)
channel.save("flow_past_cylinder_dfg_benchmark.step")
