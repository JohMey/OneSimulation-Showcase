"""Geometry of the OneSimulation example "Microchannel heat sink".

Builds one symmetry cell of a silicon microchannel heat sink with the CAD objects of OneSimulation:
half a water channel (25 um of its 50 um width, 300 um high) next to half a silicon fin, on a 100 um
silicon base, 4 mm long in the flow direction x. The cell is written to microchannel_heat_sink.step;
its faces are named after the two bodies: channel.x_min (inlet), channel.x_max (outlet),
channel.y_min (mid-plane of the channel), channel.z_max (glass cover over the channel), solid.z_min
(heated bottom), solid.y_max (mid-plane of the fin) and the silicon ends solid.x_min, solid.x_max.
The water channel alone, which marks the fluid region and carries the flow mesh, is written to
microchannel_heat_sink_channel.step. The box enclosing the cell, on which the hexahedral heat mesh
is laid, is written to microchannel_heat_sink_envelope.step.
"""

from onesim.cad.d3 import Box

um = 1e-6
length, channel, fin, base, height = 4000 * um, 25 * um, 25 * um, 100 * um, 300 * um
water = Box(length=length, width=channel, height=height, origin=(0.0, 0.0, base))
silicon = Box(length=length, width=channel + fin, height=base + height) - water
cell = water.named("channel") | silicon.named("solid")
cell.save("microchannel_heat_sink.step")
water.save("microchannel_heat_sink_channel.step")
Box(length=length, width=channel + fin, height=base + height).save(
    "microchannel_heat_sink_envelope.step"
)
