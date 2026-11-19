"""Geometry of the OneSimulation example "Laminar flow over a backward-facing step".

Builds the channel behind the step, 30 m long and 1 m high (the step height is 0.5 m), as a slab
0.1 m thick with the CAD objects of OneSimulation and writes backward_facing_step_flow.step. The
step corner sits at the origin of the face x_min: the upper half of that face is the inflow, the
lower half the vertical face of the step. The faces keep their names x_min, x_max, y_min, y_max,
z_min and z_max in the file; the example computes on the cross-section z = 0.05 m.
"""

from onesim.cad.d3 import Box

channel = Box(length=30.0, width=1.0, height=0.1, origin=(0.0, -0.5, 0.0))
channel.save("backward_facing_step_flow.step")
