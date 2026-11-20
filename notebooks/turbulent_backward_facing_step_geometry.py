"""Geometry of the OneSimulation example "Turbulent flow over a backward-facing step".

Builds two slabs, 0.1 H thick, with the CAD objects of OneSimulation (step height H = 12.7 mm):

* turbulent_backward_facing_step.step: the test section of Driver and Seegmiller (1985), 8 H high
  ahead of the step and 9 H behind it, from 4 H upstream of the step edge (at the origin) to 30 H
  behind it; faces inflow, outflow, top, floor (upstream wall), step and bottom (downstream wall);
* turbulent_backward_facing_step_inflow.step: the 8 H high inflow channel in which the incoming
  boundary layer grows, a 2 H long symmetry plane (lead) and then 100 H of floor; faces inflow,
  outflow, top, lead and floor.

The example computes on the mid-planes of both.
"""

from onesim.cad.d3 import Box

H = 0.0127
plane = (
    lambda face, axis, value: face.kind == "plane"
    and abs(face.center[axis] - value) < 1e-9
)

upstream = Box(length=4 * H, width=8 * H, height=0.1 * H, origin=(-4 * H, 0.0, 0.0))
downstream = Box(length=30 * H, width=9 * H, height=0.1 * H, origin=(0.0, -H, 0.0))
channel = upstream | downstream
channel = channel.grouped(
    inflow=channel.select(faces=lambda face: plane(face, 0, -4 * H)),
    outflow=channel.select(faces=lambda face: plane(face, 0, 30 * H)),
    top=channel.select(faces=lambda face: plane(face, 1, 8 * H)),
    floor=channel.select(faces=lambda face: plane(face, 1, 0.0) and face.center[0] < 0),
    step=channel.select(faces=lambda face: plane(face, 0, 0.0) and face.center[1] < 0),
    bottom=channel.select(faces=lambda face: plane(face, 1, -H)),
)
channel.save("turbulent_backward_facing_step.step")

lead = Box(length=2 * H, width=8 * H, height=0.1 * H, origin=(-2 * H, 0.0, 0.0))
plate = Box(length=100 * H, width=8 * H, height=0.1 * H, origin=(0.0, 0.0, 0.0))
inflow = lead | plate
inflow = inflow.grouped(
    inflow=inflow.select(faces=lambda face: plane(face, 0, -2 * H)),
    outflow=inflow.select(faces=lambda face: plane(face, 0, 100 * H)),
    top=inflow.select(faces=lambda face: plane(face, 1, 8 * H)),
    lead=inflow.select(faces=lambda face: plane(face, 1, 0.0) and face.center[0] < 0),
    floor=inflow.select(faces=lambda face: plane(face, 1, 0.0) and face.center[0] > 0),
)
inflow.save("turbulent_backward_facing_step_inflow.step")
