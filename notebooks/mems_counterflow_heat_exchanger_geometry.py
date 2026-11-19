"""Geometry of the OneSimulation example "Counterflow MEMS heat exchanger".

Builds the lower half of a stainless-steel micro heat exchanger with the CAD objects of
OneSimulation: a block 10 mm long (x), 0.9 mm wide (y) and 0.25 mm high (half of 0.5 mm, z from the
mid-plane down) with two parallel channels, each 0.3 mm wide and 0.1 mm high in the half, separated
by a 0.1 mm wall. The block is written to mems_counterflow_heat_exchanger.step; its faces are named
after the three bodies (hot.x_min, hot.x_max, cold.x_min, cold.x_max, hot.z_max, cold.z_max on the
mid-plane, steel.* for the steel). The two channels, which mark the fluid regions and carry the flow
meshes, are written to mems_counterflow_heat_exchanger_hot.step and ..._cold.step, the box enclosing
the chip, on which the heat mesh is laid, to mems_counterflow_heat_exchanger_envelope.step.
"""

from onesim.cad.d3 import Box

mm = 1e-3
length, width, height, channel, depth, wall = (
    10 * mm,
    0.9 * mm,
    0.25 * mm,
    0.3 * mm,
    0.1 * mm,
    0.1 * mm,
)
hot = Box(
    length=length, width=channel, height=depth, origin=(0.0, 0.1 * mm, height - depth)
)
cold = Box(
    length=length,
    width=channel,
    height=depth,
    origin=(0.0, 0.1 * mm + channel + wall, height - depth),
)
steel = Box(length=length, width=width, height=height) - hot - cold
(hot.named("hot") | cold.named("cold") | steel.named("steel")).save(
    "mems_counterflow_heat_exchanger.step"
)
hot.save("mems_counterflow_heat_exchanger_hot.step")
Box(length=length, width=width, height=height).save(
    "mems_counterflow_heat_exchanger_envelope.step"
)
cold.save("mems_counterflow_heat_exchanger_cold.step")
